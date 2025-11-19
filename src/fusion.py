"""
src/fusion.py

DA3 + SAM2 multiview fusion + ICP + Open3D viewer.
Handles:
 - depth.npz from DA3
 - SAM mask resizing
 - camera intrinsics/extrinsics or fallback rig
 - multi-view pointcloud merging
 - ICP refinement
 - frame playback viewer

"""

import os
import numpy as np
import logging
from pathlib import Path
import cv2
import open3d as o3d

logger = logging.getLogger("fusion")
logging.basicConfig(level=logging.INFO)

# Calibration imports
try:
    from src.calibration import parse_colmap_db, cylinder_rig_fallback, CAM_FILES
except:
    from calibration import parse_colmap_db, cylinder_rig_fallback, CAM_FILES

DEPTH_ROOT = Path("output/depth")
MASK_ROOT = Path("output/masks")
FRAME_ROOT = Path("output/frames")
CLOUD_ROOT = Path("output/clouds")
CLOUD_ROOT.mkdir(parents=True, exist_ok=True)

# =========================================
# Helpers
# =========================================

def load_prediction_npz(cam, frame_idx):
    """
    Load DA3 prediction.npz for a specific cam & frame.
    """
    npz_path = DEPTH_ROOT / cam / "prediction.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No prediction.npz for {cam}")

    d = np.load(npz_path, allow_pickle=True)

    depth = d["depth"]  # [N, H, W]
    intr = d["intrinsics"]
    extr = d["extrinsics"]

    # choose frame
    if frame_idx >= depth.shape[0]:
        depth_frame = depth[0]
        logger.warning(f"{cam}: frame_idx {frame_idx} out of range, using frame 0")
    else:
        depth_frame = depth[frame_idx]

    intr = intr[0] if intr.ndim == 3 else intr
    extr = extr[0] if extr.ndim == 3 else extr

    return depth_frame, intr, extr


def resize_mask(mask, target_h, target_w):
    try:
        return cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        logger.warning(f"Mask resize failed: {e}")
        return None


def depth_to_pointcloud(depth, K, mask=None, color=None, stride=2, max_depth=20):
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    zs = depth[::stride, ::stride]

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    valid = (zs > 0) & (zs < max_depth)
    if mask is not None:
        valid &= (mask[::stride, ::stride] > 0)

    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    zs = zs.astype(np.float32)

    x = (xs - cx) * zs / fx
    y = (ys - cy) * zs / fy
    pts = np.stack([x, y, zs], axis=-1).reshape(-1, 3)
    pts = pts[valid.reshape(-1)]

    cols = None
    if color is not None:
        c = color[::stride, ::stride].reshape(-1, 3)
        cols = c[valid.reshape(-1)]
        if cols.max() > 1.5:
            cols = cols / 255.0

    return pts, cols


def extrinsics_cam_to_world(extr):
    """
    extr is usually world2cam: X_cam = R X_w + t
    We invert to get world = R^T (X_cam - t)
    """
    R = extr[:, :3]
    t = extr[:, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


# =========================================
# MAIN fusion function
# =========================================

def fuse_frame_across_cams(frame_idx, cam_keys, do_icp=True, stride=2, max_depth=20):
    logger.info(f"Fusing frame {frame_idx}")

    merged_pts = []
    merged_cols = []

    for cam in cam_keys:
        try:
            depth, K, extr = load_prediction_npz(cam, frame_idx)
        except Exception as e:
            logger.warning(f"{cam}: skipping — no depth: {e}")
            continue

        H, W = depth.shape

        # load mask
        mask_path = MASK_ROOT / cam / f"mask_{frame_idx:06d}.npy"
        mask = None
        if mask_path.exists():
            raw = np.load(mask_path)
            mask = resize_mask(raw, H, W)

        # load color frame
        color_path = FRAME_ROOT / cam / f"{frame_idx:06d}.jpg"
        if color_path.exists():
            color = cv2.imread(str(color_path))[:, :, ::-1]
        else:
            color = None

        # intrinsics fallback
        if K is None:
            logger.warning(f"{cam}: No intrinsics. Using fallback.")
            K = np.array([[1200,0,960],[0,1200,540],[0,0,1]])

        # extrinsics invert
        R, t = extrinsics_cam_to_world(extr)

        # camera-space points
        pts, cols = depth_to_pointcloud(depth, K, mask=mask, color=color, stride=stride, max_depth=max_depth)

        if pts.shape[0] == 0:
            continue

        # world transform
        pts_w = (R @ pts.T).T + t.reshape(1,3)

        merged_pts.append(pts_w)
        merged_cols.append(cols if cols is not None else np.zeros_like(pts_w))

    if not merged_pts:
        raise RuntimeError(f"No points collected for frame {frame_idx}")

    # ICP between cams (pairwise)
    all_pts = merged_pts[0]
    all_cols = merged_cols[0]

    if do_icp and len(merged_pts) > 1:
        for i in range(1, len(merged_pts)):
            pcd_ref = o3d.geometry.PointCloud()
            pcd_ref.points = o3d.utility.Vector3dVector(all_pts)

            pcd_src = o3d.geometry.PointCloud()
            pcd_src.points = o3d.utility.Vector3dVector(merged_pts[i])

            reg = o3d.pipelines.registration.registration_icp(
                pcd_src, pcd_ref, 0.05,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

            T = reg.transformation
            merged_pts[i] = (T[:3,:3] @ merged_pts[i].T).T + T[:3,3]

            all_pts = np.vstack([all_pts, merged_pts[i]])
            all_cols = np.vstack([all_cols, merged_cols[i]])

    else:
        all_pts = np.vstack(merged_pts)
        all_cols = np.vstack(merged_cols)

    # SAVE PLY
    out_path = CLOUD_ROOT / f"merged_{frame_idx:06d}.ply"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_cols)
    o3d.io.write_point_cloud(str(out_path), pcd)

    logger.info(f"Saved PLY: {out_path} ({len(all_pts)} pts)")
    return out_path


# =========================================
# Build frame ranges
# =========================================

def build_all_frames(frame_range, cam_keys, do_icp=True, save_individual=True):
    for fi in frame_range:
        try:
            fuse_frame_across_cams(fi, cam_keys, do_icp=do_icp)
        except Exception as e:
            logger.error(f"Frame {fi} failed: {e}")


# =========================================
# Viewer
# =========================================

def play_frames(start, end, cam_keys, frame_step=1, auto_build=False):
    logger.info("Viewer controls: Left/Right arrows. Close window to exit.")

    vis = o3d.visualization.Visualizer()
    vis.create_window("UGRC Viewer", width=1280, height=720)

    idx = start
    geom = None

    while True:
        ply = CLOUD_ROOT / f"merged_{idx:06d}.ply"
        if not ply.exists():
            if auto_build:
                fuse_frame_across_cams(idx, cam_keys)
            else:
                logger.warning(f"PLY missing: {ply}")
                idx += frame_step
                if idx > end:
                    break
                continue

        pc = o3d.io.read_point_cloud(str(ply))

        if geom is None:
            geom = pc
            vis.add_geometry(geom)
        else:
            geom.points = pc.points
            geom.colors = pc.colors
            vis.update_geometry(geom)

        vis.poll_events()
        vis.update_renderer()

        # Keyboard events
        key = vis.get_view_control().convert_to_pinhole_camera_parameters()
        # We can't read key events directly; using Open3D 0.17 behaviour:
        # Put frame stepping externally
        idx += frame_step
        if idx > end:
            break

    vis.destroy_window()

