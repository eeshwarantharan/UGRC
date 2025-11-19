# src/perception.py
import os
import cv2
import numpy as np
import torch
import logging
from pathlib import Path
from .config import CAM_FILES, OUTPUT_ROOT, SAM2_ROOT, GPU_ALLOCATION
from .config import SAM_MODEL_CONFIG, SAM_MODEL_CHECKPOINT

logger = logging.getLogger("perception")
logging.basicConfig(level=logging.INFO)

MASK_CACHE_DIR = os.path.join(OUTPUT_ROOT, "masks")
Path(MASK_CACHE_DIR).mkdir(parents=True, exist_ok=True)

import subprocess

def ensure_mp4(cam_key, vid_path):
    """
    If input is .mpeg convert to mp4 using ffmpeg (idempotent).
    Returns path to mp4 file.
    """
    vid_path = Path(vid_path)
    if vid_path.suffix.lower() == ".mp4":
        return str(vid_path)

    out = vid_path.with_suffix(".mp4")
    if out.exists():
        return str(out)

    logger.info("[ffmpeg] Converting %s → %s", vid_path, out)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(vid_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        str(out)
    ]
    subprocess.run(cmd, check=True)
    return str(out)


# Try to import SAM2 APIs (local editable install or HF fallback)
SAM_AVAILABLE = False
try:
    # prefer local build API per SAM2 README
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    logger.info("Local SAM2 import OK")
    SAM_AVAILABLE = True
except Exception as e:
    logger.warning("Local sam2 import failed (%s). Trying HF import API...", e)
    try:
        # HF convenience loader if installed
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        SAM_AVAILABLE = True
    except Exception:
        SAM_AVAILABLE = False
        logger.warning("SAM2 not available. Will use optical-flow fallback for propagation.")


def open_video_capture(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {path}")
    return cap


def mid_stream_initialization(cam_key="cam01"):
    """
    Open video, seek to middle frame, let user draw a bounding box (click-drag).
    Returns mid_frame_idx, bbox=(x,y,w,h)
    """
    video_path = CAM_FILES[cam_key]
    video_path = ensure_mp4(cam_key, video_path)
    cap = open_video_capture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n == 0:
        cap.release()
        raise RuntimeError("Video has 0 frames")
    mid = max(0, n // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to read mid frame for init.")

    clone = frame.copy()
    bbox = []
    drawing = False

    def mouse_cb(evt, x, y, flags, param):
        nonlocal bbox, clone, drawing
        if evt == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            bbox = [(x,y)]
        elif evt == cv2.EVENT_MOUSEMOVE and drawing:
            tmp = clone.copy()
            cv2.rectangle(tmp, bbox[0], (x,y), (0,255,0), 2)
            cv2.imshow("Initialize - draw a box then press 's'", tmp)
        elif evt == cv2.EVENT_LBUTTONUP:
            drawing = False
            bbox.append((x,y))
            cv2.rectangle(clone, bbox[0], bbox[1], (0,255,0), 2)
            cv2.imshow("Initialize - draw a box then press 's'", clone)

    win = "Initialize - draw a box then press 's'"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_cb)
    while True:
        display = clone.copy()
        if len(bbox) == 2:
            cv2.rectangle(display, bbox[0], bbox[1], (0,255,0), 2)
        cv2.imshow(win, display)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("s") and len(bbox) == 2:
            break
        if k == ord("q"):
            cv2.destroyAllWindows()
            cap.release()
            raise KeyboardInterrupt("User aborted initialization")
    cv2.destroyAllWindows()
    cap.release()
    x1,y1 = bbox[0]; x2,y2 = bbox[1]
    x,y,w,h = min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)
    logger.info("Init bbox: %s at frame %d", (x,y,w,h), mid)
    return mid, (x,y,w,h)


def save_mask(cam_key, frame_idx, mask):
    d = Path(MASK_CACHE_DIR) / cam_key
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"mask_{frame_idx:06d}.npy", (mask.astype(np.uint8)))


def load_mask(cam_key, frame_idx):
    p = Path(MASK_CACHE_DIR) / cam_key / f"mask_{frame_idx:06d}.npy"
    if p.exists():
        m = np.load(p)
        # ensure single-channel
        if m.ndim == 3:
            m = m[..., 0]
        return (m > 0).astype(np.uint8)
    return None


def propagate_with_sam(cam_key, init_frame_idx, init_bbox):
    """
    Use SAM2VideoPredictor API to get masks for whole video and save them.
    This function supports both local build_sam2_video_predictor and SAM2VideoPredictor.from_pretrained.
    """
    # ensure mp4
    video_path = ensure_mp4(cam_key, CAM_FILES[cam_key])

    # Build predictor
    predictor = None
    try:
        if 'build_sam2_video_predictor' in globals():
            # pass strings for config and checkpoint (these come from config.py)
            model_cfg = SAM_MODEL_CONFIG
            model_ckpt = SAM_MODEL_CHECKPOINT
            predictor = build_sam2_video_predictor(model_cfg, model_ckpt)
        else:
            predictor = SAM2VideoPredictor.from_pretrained(SAM_MODEL_CHECKPOINT)
    except Exception as e:
        logger.exception("Failed to build SAM2 predictor: %s", e)
        raise

    logger.info("SAM2 predictor built successfully.")

    # SAM2 init_state accepts a video path or list of frames depending on implementation.
    # Prefer passing path (some builds expect that).
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path)
    except Exception as e:
        # fallback: read frames into list (RGB)
        logger.info("SAM2 init_state(path) failed, reading frames into memory (%s)", e)
        frames = []
        cap = open_video_capture(video_path)
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            frames.append(fr[:, :, ::-1])  # BGR->RGB
        cap.release()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(frames)

    # Build prompt: SAM2 add_new_points_or_box requires obj_id param in local API (use 0)
    x, y, w, h = init_bbox
    box = [int(x), int(y), int(x + w), int(y + h)]
    init_frame = int(init_frame_idx)
    try:
        # local API signature: add_new_points_or_box(self, state, obj_id, frame_idx, box=..., points=...)
        obj_id = 0
        frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, obj_id, init_frame, box=box)
    except TypeError:
        # try alternate signature (frame_idx positional)
        try:
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, init_frame, box=box)
        except Exception as e:
            logger.exception("Unsupported add_new_points_or_box signature: %s", e)
            raise

    # save init mask if present
    if masks is not None and len(masks) > 0:
        m = masks[0]
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        m = ((m > 0.5).astype(np.uint8)) * 255
        save_mask(cam_key, init_frame, m)

    # propagate
    try:
        for frame_idx, obj_ids, frame_masks in predictor.propagate_in_video(state):
            if frame_masks is None or len(frame_masks) == 0:
                continue
            # assume single tracked object (object index 0)
            m = frame_masks[0]
            if isinstance(m, torch.Tensor):
                m = m.detach().cpu().numpy()
            m = ((m > 0.5).astype(np.uint8)) * 255
            save_mask(cam_key, frame_idx, m)
    except Exception as e:
        logger.exception("SAM propagation failed: %s", e)
        raise

    logger.info("SAM propagation finished for %s", cam_key)


def propagate_with_optflow(cam_key, init_frame_idx, init_bbox):
    """
    Optical-flow fallback when SAM2 is unavailable.
    Uses Farneback flow + warp of masks. Crude but sometimes useful.
    """
    logger.warning("[%s] Using optical-flow fallback (low accuracy).", cam_key)

    video_path = ensure_mp4(cam_key, CAM_FILES[cam_key])
    cap = open_video_capture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    for _ in range(n):
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(frames) == 0:
        raise RuntimeError("No frames read for optical-flow fallback")

    x, y, w, h = init_bbox
    init_mask = np.zeros_like(frames[init_frame_idx], dtype=np.uint8)
    init_mask[y:y+h, x:x+w] = 255
    save_mask(cam_key, init_frame_idx, init_mask)

    # forward
    prev = frames[init_frame_idx]
    prev_mask = init_mask
    for i in range(init_frame_idx, len(frames) - 1):
        next_f = frames[i+1]
        flow = cv2.calcOpticalFlowFarneback(
            prev, next_f, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        h_, w_ = prev_mask.shape
        grid_x, grid_y = np.meshgrid(np.arange(w_), np.arange(h_))
        map_x = (grid_x + flow[...,0]).astype(np.float32)
        map_y = (grid_y + flow[...,1]).astype(np.float32)
        next_mask = cv2.remap(prev_mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        save_mask(cam_key, i+1, next_mask)
        prev = next_f
        prev_mask = next_mask

    # backward
    prev = frames[init_frame_idx]
    prev_mask = init_mask
    for i in reversed(range(1, init_frame_idx)):
        next_f = frames[i-1]
        flow = cv2.calcOpticalFlowFarneback(
            prev, next_f, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        h_, w_ = prev_mask.shape
        grid_x, grid_y = np.meshgrid(np.arange(w_), np.arange(h_))
        map_x = (grid_x + flow[...,0]).astype(np.float32)
        map_y = (grid_y + flow[...,1]).astype(np.float32)
        next_mask = cv2.remap(prev_mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        save_mask(cam_key, i-1, next_mask)
        prev = next_f
        prev_mask = next_mask

    logger.info("[%s] Optical-flow fallback propagation done.", cam_key)


def track_person_masks(cam_key, force_recompute=False):
    """
    Main public method. Tracks person masks for cam_key and saves to OUTPUT_ROOT/masks/<cam_key>/
    """
    cam_cache = Path(MASK_CACHE_DIR) / cam_key
    if cam_cache.exists() and not force_recompute:
        logger.info("[%s] Masks already cached → skipping perception.", cam_key)
        return cam_cache

    if cam_cache.exists():
        for p in cam_cache.glob("*.npy"):
            p.unlink()

    init_frame_idx, init_bbox = mid_stream_initialization(cam_key)

    if SAM_AVAILABLE:
        logger.info("[%s] Using SAM2 for segmentation.", cam_key)
        propagate_with_sam(cam_key, init_frame_idx, init_bbox)
    else:
        logger.warning("[%s] SAM2 unavailable → using optical-flow fallback.", cam_key)
        propagate_with_optflow(cam_key, init_frame_idx, init_bbox)

    return cam_cache


if __name__ == "__main__":
    # manual test: run SAM2/optflow on a single camera
    track_person_masks("cam01", force_recompute=True)
    print("Done.")

