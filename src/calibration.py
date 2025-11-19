# src/calibration.py
import os
import subprocess
import logging
from pathlib import Path
import sqlite3
import numpy as np

from .config import IMAGES_OUTPUT, CAM_FILES, DIM_MODULE_NAME, OUTPUT_ROOT, COLMAP_DB, FORCE_RIG_FALLBACK

logger = logging.getLogger("calibration")
logging.basicConfig(level=logging.INFO)

def ensure_dirs():
    Path(IMAGES_OUTPUT).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

def extract_first_frames():
    ensure_dirs()
    for cam_key, vid in CAM_FILES.items():
        out = Path(IMAGES_OUTPUT) / f"{cam_key}.jpg"
        if out.exists():
            logger.info(f"{out} exists, skip ffmpeg.")
            continue
        if not Path(vid).exists():
            logger.warning(f"Video {vid} not found for {cam_key}")
            continue
        cmd = ["ffmpeg", "-y", "-i", str(vid), "-frames:v", "1", "-q:v", "2", str(out)]
        logger.info("Running: " + " ".join(cmd))
        subprocess.run(cmd, check=True)

def run_dim_via_api(timeout=None):
    """
    Use deep_image_matching python API to run feature extraction, matching and export to COLMAP db.
    Follows the pattern in the DIM notebooks you provided.
    """
    try:
        import deep_image_matching as dim
        from deep_image_matching.utils import OutputCapture
    except Exception as e:
        logger.exception("deep_image_matching import failed. Try running as module fallback.")
        raise

    params = {
        "dir": str(Path(IMAGES_OUTPUT).parent),  # DIM expects 'dir' pointing to project with images in images/
        "pipeline": "superpoint+lightglue",
        "strategy": "matching_lowres",
        "quality": "high",
        "tiling": "none",
        "skip_reconstruction": False,
        "force": True,
        "camera_options": None,  # you can provide cameras.yaml path here
        "openmvg": None,
        "verbose": False,
    }
    logger.info("Building DIM config...")
    config = dim.Config(params)

    logger.info("Running DIM ImageMatcher...")
    matcher = dim.ImageMatcher(config)
    feature_path, match_path = matcher.run()

    logger.info("Exporting to COLMAP database...")
    db_path = Path(config.general["output_dir"]) / "database.db"
    dim.io.export_to_colmap(
        img_dir=config.general["image_dir"],
        feature_path=feature_path,
        match_path=match_path,
        database_path=db_path,
        camera_config_path=config.general.get("camera_options", None),
    )

    # Try reconstruction via DIM wrapper which uses pycolmap when available
    try:
        logger.info("Running incremental reconstruction (pycolmap) via DIM wrapper...")
        reconstruction = dim.reconstruction.incremental_reconstruction(
            database_path=db_path,
            image_dir=config.general["image_dir"],
            sfm_dir=config.general["output_dir"],
            refine_intrinsics=False,
            reconstruction_options=None,
        )
        logger.info("Reconstruction finished.")
    except Exception as e:
        logger.warning("Reconstruction via DIM failed: %s", e)
        reconstruction = None

    return db_path if db_path.exists() else None

def run_dim_module_cli():
    """
    Fallback: call python -m deep_image_matching as a subprocess.
    This keeps with your 'run as module' constraint.
    """
    try:
        cmd = ["python", "-m", DIM_MODULE_NAME, "--input", IMAGES_OUTPUT, "--output", str(Path(OUTPUT_ROOT))]
        logger.info("Running DIM module CLI: " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        # expect database.db in output
        db_candidate = Path(OUTPUT_ROOT) / "database.db"
        if db_candidate.exists():
            return db_candidate
    except Exception:
        logger.exception("DIM module CLI failed.")
    return None

def parse_colmap_db(db_path):
    """
    Parse the COLMAP SQLite DB and return dict: {image_name: {"K":K, "R":R, "t":t}}
    Attempts various common COLMAP DB schemas.
    """
    if db_path is None or not Path(db_path).exists():
        logger.warning("No COLMAP DB to parse.")
        return {}

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # load cameras table
    cams = {}
    try:
        for row in cur.execute("SELECT camera_id, model, width, height, params FROM cameras;"):
            camera_id, model, width, height, params = row
            # params may be blob -> try to decode
            try:
                if isinstance(params, (bytes, bytearray)):
                    # COLMAP stores params as blob of doubles; use sqlite3 built-in to get bytes -> decode floats:
                    # naive fallback: try to eval string
                    params = np.frombuffer(params, dtype=np.float64).tolist()
                elif isinstance(params, str):
                    params = list(map(float, params.split()))
            except Exception:
                try:
                    params = list(map(float, params))
                except Exception:
                    params = []
            cams[camera_id] = {"model": model, "width": width, "height": height, "params": params}
    except Exception:
        logger.exception("Failed to read cameras table.")

    results = {}
    # try different images table schemas
    candidates = []
    for q in [
        "SELECT image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz FROM images;",
        "SELECT image_id, name, camera_id, qw, qx, qy, qz, tx, ty, tz FROM images;",
        "SELECT image_id, name, camera_id, qx, qy, qz, qw, tx, ty, tz FROM images;",
    ]:
        try:
            for row in cur.execute(q):
                candidates.append(row)
            if candidates:
                break
        except Exception:
            continue

    for row in candidates:
        if len(row) == 10:
            image_id = row[0]; name = row[1]; camera_id = row[2]
            # identify quat ordering
            rest = row[3:]
            if len(rest) == 7:
                # prior_qw... pattern (rare)
                qw, qx, qy, qz, tx, ty, tz = rest
            elif len(rest) == 6:
                qw, qx, qy, qz, tx, ty = rest  # malformed
            else:
                qw, qx, qy, qz, tx, ty, tz = 1,0,0,0,0,0,0
        else:
            continue
        q = np.array([qw, qx, qy, qz], dtype=float)
        q = q / np.linalg.norm(q)
        qw, qx, qy, qz = q.tolist()
        R = quat_to_rotmat(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=float).reshape(3,1)
        cam = cams.get(camera_id, {})
        K = parse_camera_params(cam)
        results[name] = {"K": K, "R": R, "t": t}
    conn.close()
    return results

def quat_to_rotmat(qw, qx, qy, qz):
    R = np.zeros((3,3), dtype=float)
    R[0,0] = 1 - 2*(qy*qy + qz*qz)
    R[0,1] = 2*(qx*qy - qz*qw)
    R[0,2] = 2*(qx*qz + qy*qw)
    R[1,0] = 2*(qx*qy + qz*qw)
    R[1,1] = 1 - 2*(qx*qx + qz*qz)
    R[1,2] = 2*(qy*qz - qx*qw)
    R[2,0] = 2*(qx*qz - qy*qw)
    R[2,1] = 2*(qy*qz + qx*qw)
    R[2,2] = 1 - 2*(qx*qx + qy*qy)
    return R

def parse_camera_params(cam_row):
    if not cam_row:
        return None
    params = cam_row.get("params", None)
    if not params:
        return None
    try:
        if isinstance(params, (list, tuple)):
            params = list(map(float, params))
        elif isinstance(params, str):
            params = list(map(float, params.split()))
    except Exception:
        params = []
    if len(params) >= 4:
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)
        return K
    if len(params) >= 3:
        fx, cx, cy = params[0], params[1], params[2]
        K = np.array([[fx,0,cx],[0,fx,cy],[0,0,1]], dtype=float)
        return K
    return None

def cylinder_rig_fallback():
    logger.warning("Using cylinder rig fallback.")
    names = list(CAM_FILES.keys())
    n = len(names)
    cams = {}
    radius = 3.0
    for i, name in enumerate(names):
        theta = 2*np.pi * i / n
        pos = np.array([radius*np.cos(theta), radius*np.sin(theta), 1.6])
        z = -pos / np.linalg.norm(pos)
        up = np.array([0,0,1.0])
        x = np.cross(up, z); x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.vstack([x,y,z]).T
        t = -R @ pos.reshape(3,1)
        K = np.array([[1200,0,640],[0,1200,360],[0,0,1]], dtype=float)
        cams[f"{name}.jpg"] = {"K":K, "R":R, "t":t}
    return cams

def calibrate_all():
    extract_first_frames()
    db = None
    try:
        db = run_dim_via_api()
    except Exception:
        logger.exception("DIM API run failed; trying CLI fallback.")
        db = run_dim_module_cli()
    cams = {}
    if db:
        cams = parse_colmap_db(db)

    if FORCE_RIG_FALLBACK or not cams:
    	logger.warning("Forcing cylinder rig fallback.")
    	cams = cylinder_rig_fallback()

    return cams

if __name__ == "__main__":
    cams = calibrate_all()
    print("Cam keys:", list(cams.keys()))

