"""
geometry.py

Multiview mask fusion utilities.

Strategy implemented:
  - Visual hull via occupancy testing on a 3D regular grid:
      * Build 3D sampling grid bounded by camera centers ± margin
      * For each sample point, project into each camera
      * If projected pixel is inside that cam's binary mask for *all* cameras considered,
        the point survives hull carving.
  - Returns surviving 3D points and a centroid.

Notes:
  - This is intentionally simple and robust. For better quality you can:
    * increase grid_resolution
    * use adaptive / octree sampling
    * perform Poisson surface reconstruction on surviving points
"""

from pathlib import Path
import numpy as np
import logging
from .config import OUTPUT_ROOT

logger = logging.getLogger("geometry")
logging.basicConfig(level=logging.INFO)


def project_point(K, R, t, points):
    """
    Project Nx3 points (world coords) to image pixel coordinates using K, R, t.
    R: 3x3, t: 3x1
    Returns: Nx2 pixel coords and Nx bool for valid (in front of camera)
    """
    # points: (N,3)
    # camera coords: Xc = R @ Xw + t
    Xc = (R @ points.T) + t.reshape(3, 1)  # (3,N)
    Xc = Xc.T  # (N,3)
    z = Xc[:, 2:3]
    # avoid division by zero
    eps = 1e-8
    z_safe = np.where(z <= eps, eps, z)
    uv = (K @ Xc.T).T  # (N,3) but expects homogeneous
    u = uv[:, 0] / uv[:, 2]
    v = uv[:, 1] / uv[:, 2]
    pix = np.stack([u, v], axis=1)
    in_front = (Xc[:, 2] > 1e-4)
    return pix, in_front


def save_points_as_ply(points: np.ndarray, outpath):
    """
    Save Nx3 float points as simple PLY (no colors).
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(outpath, "w") as f:
        f.write(header)
        for p in pts:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def fuse_multiview_masks(cams, cam_keys, masks, grid_resolution=50, margin=1.0):
    """
    cams: mapping image_name -> {"K":K, "R":R, "t":t}
          note: keys in cams are e.g. "cam01.jpg" — cam_keys passed must match those keys (or we map)
    cam_keys: list of keys matching the 'masks' dict and matching entries of 'cams' (or camera names)
    masks: dict cam_key -> 2D numpy array (binary masks: non-zero foreground)
    grid_resolution: number of voxels per axis (cubic grid)
    margin: float meters to expand bounding box around camera centers

    returns: Nx3 numpy array of 3D points that survive the hull carving
    """
    # first build a camera-to-pose mapping. We accept cam_keys like 'cam01' or 'cam01.jpg'.
    cam_pose_map = {}
    for k in cam_keys:
        # attempt several key forms
        search_keys = [k, f"{k}.jpg", f"{k}.png"]
        found = None
        for sk in search_keys:
            if sk in cams:
                found = sk
                break
        if found is None:
            logger.warning("No cam pose for %s (keys tried: %s)", k, search_keys)
            continue
        cam_pose_map[k] = cams[found]

    if len(cam_pose_map) < 2:
        logger.warning("Need >=2 calibrated cameras for fusion; got %d", len(cam_pose_map))
        return np.zeros((0, 3), dtype=float)

    # compute approximate bounding box for sampling: use camera centers
    cam_centers = []
    for k, info in cam_pose_map.items():
        K = np.array(info["K"], dtype=float)
        R = np.array(info["R"], dtype=float)
        t = np.array(info["t"], dtype=float).reshape(3)
        # camera center in world coordinates: C = -R^T * t
        C = -R.T @ t
        cam_centers.append(C)
    cam_centers = np.stack(cam_centers, axis=0)
    center = np.mean(cam_centers, axis=0)
    maxdist = np.max(np.linalg.norm(cam_centers - center[None, :], axis=1)) + margin

    # grid bounds cube centered at center with side = 2*maxdist
    half = maxdist
    xs = np.linspace(center[0] - half, center[0] + half, grid_resolution)
    ys = np.linspace(center[1] - half, center[1] + half, grid_resolution)
    zs = np.linspace(center[2] - half, center[2] + half, grid_resolution)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")
    samples = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (M,3), M = grid_resolution^3

    logger.debug("Sampling %d points for visual hull (grid_resolution=%d)", samples.shape[0], grid_resolution)
    # For each camera, project the whole sample set and check if projection is inside mask.
    survivors = np.ones((samples.shape[0],), dtype=bool)

    for cam_key, mask in masks.items():
        if cam_key not in cam_pose_map:
            logger.warning("Camera %s not in pose map; skipping it for hull carving", cam_key)
            continue
        info = cam_pose_map[cam_key]
        K = np.array(info["K"], dtype=float)
        R = np.array(info["R"], dtype=float)
        t = np.array(info["t"], dtype=float).reshape(3, 1)

        h, w = mask.shape[:2]
        pix, in_front = project_point(K, R, t, samples)
        u = np.round(pix[:, 0]).astype(int)
        v = np.round(pix[:, 1]).astype(int)

        # in image bounds
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (in_front)
        # if a point projects inside the image, check mask value
        ok = np.zeros_like(inside)
        inside_idx = np.nonzero(inside)[0]
        if inside_idx.size:
            # read mask values at those pixels
            # mask may be uint8 with 0 background, >0 foreground
            vals = mask[v[inside_idx], u[inside_idx]]  # note: mask is (h,w) and indexed [row,col]
            ok_inside = (vals > 0)
            ok[inside_idx] = ok_inside

        # For visual hull we typically require that the point projects to foreground in *all* cameras.
        # Here we update survivors = survivors & ok
        survivors = survivors & ok

        # Early break if nothing remains
        if not np.any(survivors):
            logger.debug("No survivors left after processing camera %s", cam_key)
            return np.zeros((0, 3), dtype=float)

    pts = samples[survivors]
    logger.info("Visual hull produced %d points", pts.shape[0])
    return pts


# Additional helper: compute centroid from pts
def compute_centroid(points):
    if points is None or len(points) == 0:
        return None
    return np.mean(points, axis=0)

