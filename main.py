#!/usr/bin/env python3
"""
main.py — One-shot UGRC pipeline runner.

Chains:
  1) calibration (if available)
  2) run_sam_all (skipped if masks exist)
  3) DepthAnything3 inference for each camera (safe, per-frame)
  4) fusion.build_all_frames (ICP)
  5) GUI viewer (gui_viewer.run_gui)

Usage:
  python main.py --start 170 --end 190 --cams cam01 cam02 cam03 cam04 --gui
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Ensure local DA3 code in path
sys.path.append("models/depth_anything_v3/src")

# UGRC modules
from src.config import CAM_FILES
try:
    from src import calibration
except Exception:
    calibration = None
from src import fusion

# DepthAnything3
from depth_anything_3.api import DepthAnything3

# utility
import time

# Directories
OUTPUT_FRAMES = Path("output/frames")
OUTPUT_DEPTH = Path("output/depth")
OUTPUT_MASKS = Path("output/masks")
OUTPUT_CLOUDS = Path("output/clouds")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def exists_sam_masks_for_all(cams, start, end):
    """Return True if masks for all cams & at least start..end exist (coarse check)."""
    for cam in cams:
        cam_dir = OUTPUT_MASKS / cam
        if not cam_dir.exists():
            return False
        # Check at least one mask (coarse)
        found = any(cam_dir.glob("mask_*.npy"))
        if not found:
            return False
    return True

# ---------------------------------------------------------------------------
# STEP 1 — Calibration
# ---------------------------------------------------------------------------
def run_calibration():
    logger.info("=== STEP 1: Calibration ===")
    if calibration is None:
        logger.warning("Calibration module not importable — skipping and using fallback.")
        return False
    if hasattr(calibration, "main"):
        try:
            calibration.main()
            logger.info("Calibration finished.")
            return True
        except Exception as e:
            logger.warning("Calibration failed: %s — using fallback.", e)
            return False
    else:
        logger.warning("Calibration.main() not found — using fallback.")
        return False

# ---------------------------------------------------------------------------
# STEP 2 — SAM2 masks (wrapper to run existing script)
# ---------------------------------------------------------------------------
def run_sam_all_if_needed(cams, start, end, force=False):
    """
    call existing script run_sam_all.py (assumed present in repo root).
    If masks already exist (coarse test) we skip unless force=True.
    """
    if not force and exists_sam_masks_for_all(cams, start, end):
        logger.info("✔ SAM2 masks already exist — skipping.")
        return

    run_script = Path("run_sam_all.py")
    if run_script.exists():
        logger.info("=== STEP 2: Running SAM2 mask propagation for all cams ===")
        import subprocess
        try:
            subprocess.run([sys.executable, str(run_script)], check=True)
            logger.info("SAM2 run completed.")
        except subprocess.CalledProcessError as e:
            logger.error("SAM2 run failed: %s", e)
            raise
    else:
        logger.warning("run_sam_all.py not found — skipping SAM stage (expect masks in output/masks).")

# ---------------------------------------------------------------------------
# STEP 3 — DepthAnything3 inference (safe, per-frame)
# ---------------------------------------------------------------------------
def run_da3_all(cams, start, end, device="cuda", force=False, model_name="depth-anything/da3nested-giant-large"):
    """
    Run DA3 inference per-camera, per-frame (safe):
      - uses DepthAnything3.from_pretrained()
      - saves outputs to output/depth/<cam>/prediction_{frame:06d}.npz (or prediction.npz)
      - skips frames already present (unless force=True)
      - tries FP16 on OOM
    """
    logger.info("=== STEP 3: DepthAnything3 inference ===")
    OUTPUT_DEPTH.mkdir(parents=True, exist_ok=True)

    # Load model (may download)
    logger.info("Loading DA3 model: %s", model_name)
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)

    # recommended: allow half precision if available
    use_fp16 = False
    try:
        # try switching to fp16 to save memory if GPU low
        if device.startswith("cuda"):
            model.half()
            use_fp16 = True
            logger.info("Using fp16 (model.half()) to reduce memory usage.")
    except Exception:
        use_fp16 = False

    # process each cam
    for cam in cams:
        logger.info("Processing depth for %s", cam)
        frame_dir = OUTPUT_FRAMES / cam
        if not frame_dir.exists():
            logger.error("Frame directory missing for %s -> skipping", cam)
            continue

        images = sorted(str(p) for p in frame_dir.glob("*.jpg"))
        if not images:
            logger.warning("No frames found for %s -> skipping", cam)
            continue

        out_dir = OUTPUT_DEPTH / cam
        out_dir.mkdir(parents=True, exist_ok=True)

        # Option: if single big export is desired, you could call model.inference(images, export_format="npz")
        # but we run per-frame to be robust.
        total = len(images)
        logger.info("Total frames: %d", total)
        for i, img_path in enumerate(images, start=1):
            try:
                frame_id = int(Path(img_path).stem)
            except Exception:
                # fallback: use index
                frame_id = i

            npz_path = out_dir / f"prediction_{frame_id:06d}.npz"
            # some DA3 variants produce prediction.npz per directory — treat either as valid
            alt_npz = out_dir / "prediction.npz"
            if not force and (npz_path.exists() or alt_npz.exists()):
                logger.debug("[%s] Skipping frame %d (exists)", cam, frame_id)
                continue

            logger.info("[%s] DA3 frame %d/%d: %s", cam, i, total, Path(img_path).name)
            try:
                # Call inference for single image — export npz
                model.inference([img_path], export_dir=str(out_dir), export_format="npz")
                # model writes prediction.npz; if it wrote prediction.npz rename to per-frame file if needed:
                pn = out_dir / "prediction.npz"
                if pn.exists():
                    # If single image -> PN contains one frame; rename (safe)
                    try:
                        dst = out_dir / f"prediction_{frame_id:06d}.npz"
                        pn.rename(dst)
                    except Exception:
                        # keep as prediction.npz if rename fails
                        pass
            except RuntimeError as e:
                # catch CUDA OOM or other runtime errors; try fp16 fallback or smaller model
                logger.warning("[%s] DA3 runtime error on frame %d: %s", cam, frame_id, e)
                # if not already using fp16, try switching
                if (not use_fp16) and device.startswith("cuda"):
                    try:
                        logger.info("Retrying with fp16...")
                        model.half()
                        use_fp16 = True
                        model.inference([img_path], export_dir=str(out_dir), export_format="npz")
                        pn = out_dir / "prediction.npz"
                        if pn.exists():
                            pn.rename(out_dir / f"prediction_{frame_id:06d}.npz")
                        continue
                    except Exception as e2:
                        logger.warning("fp16 retry failed: %s", e2)

                logger.error("[%s] DA3 frame %d failed irrecoverably: %s", cam, frame_id, e)
            except Exception as ex:
                logger.exception("[%s] DA3 frame %d unexpected error: %s", cam, frame_id, ex)

        logger.info("DA3 done for %s → %s", cam, out_dir)

# ---------------------------------------------------------------------------
# STEP 4 — fusion
# ---------------------------------------------------------------------------
def build_fusion(start, end, cams, step=1, do_icp=True):
    logger.info("=== STEP 4: Fusion (multi-view + ICP) ===")
    frame_range = list(range(start, end + 1, step))
    fusion.build_all_frames(frame_range=frame_range, cam_keys=cams, do_icp=do_icp)
    logger.info("Merged clouds saved to output/clouds/")

# ---------------------------------------------------------------------------
# STEP 5 — GUI viewer
# ---------------------------------------------------------------------------
def run_gui_viewer(start, end, cams):
    logger.info("=== STEP 5: GUI Viewer ===")
    try:
        import gui_viewer
    except Exception:
        logger.error("gui_viewer.py not found or import failed. Please add gui_viewer.py next to main.py")
        return
    gui_viewer.run_gui(start=start, end=end, cam_keys=cams)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=10)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--cams", nargs="+", default=None)
    p.add_argument("--no-view", action="store_true")
    p.add_argument("--force-da3", action="store_true")
    p.add_argument("--force-sam", action="store_true")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()

def main():
    args = parse_args()
    cams = args.cams if args.cams else sorted(list(CAM_FILES.keys()))
    logger.info("Pipeline cameras: %s", cams)
    logger.info("Frames: %d → %d, step=%d", args.start, args.end, args.step)

    # Step 1
    _ = run_calibration()

    # Step 2: SAM2 (skip if masks exist unless forced)
    run_sam_all_if_needed(cams, args.start, args.end, force=args.force_sam)

    # Step 3: DA3
    run_da3_all(cams, args.start, args.end, device=args.device, force=args.force_da3)

    # Step 4: fusion (ICP)
    build_fusion(args.start, args.end, cams, step=args.step, do_icp=True)

    # Step 5: GUI viewer
    if args.gui and not args.no_view:
        run_gui_viewer(args.start, args.end, cams)
    else:
        logger.info("GUI viewer skipped.")

if __name__ == "__main__":
    main()

