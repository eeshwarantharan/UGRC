# src/depth_da3.py
"""
Wrap Depth-Anything-3 (DA3) inference and save per-camera per-frame depth maps,
intrinsics and extrinsics. Handles multiple DA3 API variants gracefully.

Usage:
  from src.depth_da3 import run_da3_on_camera
  run_da3_on_camera("cam01", frames=[178,179], out_dir="output/depth")
"""
import os
import numpy as np
from pathlib import Path
import logging
import cv2

logger = logging.getLogger("depth_da3")
logging.basicConfig(level=logging.INFO)

# try DA3 import (local repo or pip)
try:
    # prefer local repo path: models/depth_anything_v3/src is expected in project root
    from depth_anything_3.api import DepthAnything3
    DA3_AVAILABLE = True
except Exception as e:
    logger.warning("Depth Anything 3 import failed: %s", e)
    DA3_AVAILABLE = False

# default output folder
DEFAULT_OUT = Path("output/depth")
DEFAULT_OUT.mkdir(parents=True, exist_ok=True)

def _load_image(img_path):
    im = cv2.imread(str(img_path))
    if im is None:
        raise FileNotFoundError(f"Cannot open image: {img_path}")
    return im[:,:,::-1]  # BGR -> RGB

def _ensure_model(device="cuda", model_name_or_path=None):
    if not DA3_AVAILABLE:
        raise RuntimeError("Depth Anything 3 is not available in this environment.")
    # two common patterns: from_pretrained or direct constructor
    try:
        if model_name_or_path:
            model = DepthAnything3.from_pretrained(model_name_or_path)
        else:
            model = DepthAnything3()  # local default
    except Exception:
        # fallback: direct constructor
        model = DepthAnything3()
    model = model.to(device=device)
    return model

def run_da3_on_images(image_list, out_dir=None, device="cuda", model_name_or_path=None):
    """
    Run DA3 on a list of images (paths or numpy arrays). Returns prediction object.
    Saves depth, conf, intrinsics, extrinsics to out_dir.
    """
    if out_dir is None:
        out_dir = DEFAULT_OUT
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _ensure_model(device=device, model_name_or_path=model_name_or_path)
    # DA3 API differences:
    # - model.inference(images, export_dir=..., export_format=...) -> returns Prediction with attributes
    # - some versions provide .inference; older may have 'inference' method name but same behavior
    imgs_for_da3 = []
    for im in image_list:
        if isinstance(im, (str, Path)):
            imgs_for_da3.append(_load_image(im))
        else:
            imgs_for_da3.append(im)

    # Prefer inference() with minimal export (we don't require export_format)
    try:
        pred = model.inference(imgs_for_da3, export_dir=str(out_dir), export_format="npz")
    except TypeError:
        # Some variants require no export_format or different args: try without export_format
        pred = model.inference(imgs_for_da3, export_dir=str(out_dir))
    except Exception as e:
        # final fallback: try inference returning dict-like object
        try:
            pred = model.inference(imgs_for_da3)
        except Exception as e2:
            raise RuntimeError("DA3 inference failed: %s / %s" % (e, e2))

    # pred usually has .depth, .conf, .extrinsics, .intrinsics
    # Normalize shapes to [N,H,W]
    depth = getattr(pred, "depth", None)
    conf = getattr(pred, "conf", None)
    intrinsics = getattr(pred, "intrinsics", None)
    extrinsics = getattr(pred, "extrinsics", None)

    # Save numpy arrays
    if depth is not None:
        depth_np = np.array(depth)  # ensure numpy
        np.save(out_dir / "depth.npy", depth_np)
        logger.info("Saved depth: %s", out_dir / "depth.npy")
    if conf is not None:
        np.save(out_dir / "conf.npy", np.array(conf))
    if intrinsics is not None:
        np.save(out_dir / "intrinsics.npy", np.array(intrinsics))
    if extrinsics is not None:
        np.save(out_dir / "extrinsics.npy", np.array(extrinsics))

    # Also save per-frame npz for convenience
    try:
        np.savez(out_dir / "prediction.npz",
                 depth=depth_np,
                 conf=np.array(conf) if conf is not None else None,
                 intrinsics=np.array(intrinsics) if intrinsics is not None else None,
                 extrinsics=np.array(extrinsics) if extrinsics is not None else None)
    except Exception:
        pass

    return pred

def run_da3_on_camera(cam_key, frames=None, frames_dir=None, out_dir=None, device="cuda", model_name_or_path=None):
    """
    Convenience: run DA3 on one camera. Expects frames extracted previously in:
      frames_dir (e.g., output/frames/<cam_key>/000178.jpg)
    frames: list of frame indices to process (if None, process all jpgs in frames_dir)
    Saves outputs into out_dir/<cam_key>/
    """
    if frames_dir is None:
        frames_dir = Path("output/frames") / cam_key
    frames_dir = Path(frames_dir)
    if out_dir is None:
        out_dir = Path(DEFAULT_OUT) / cam_key
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not frames_dir.exists():
        # maybe images are under output/images with single jpg per cam (first frame)
        logger.warning("Frames dir %s missing; falling back to output/images (single image)", frames_dir)
        single = Path("output/images") / f"{cam_key}.jpg"
        if not single.exists():
            raise FileNotFoundError("No frames found for camera %s" % cam_key)
        frames_list = [single]
    else:
        all_imgs = sorted(frames_dir.glob("*.jpg"))
        if frames is None:
            frames_list = all_imgs
        else:
            frames_list = [frames_dir / f"{i:06d}.jpg" if isinstance(i, int) else frames_dir / i for i in frames]

    # run DA3 on the chosen images
    pred = run_da3_on_images(frames_list, out_dir=out_dir, device=device, model_name_or_path=model_name_or_path)
    return out_dir

