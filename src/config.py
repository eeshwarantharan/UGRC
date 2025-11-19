# src/config.py
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
MODELS_ROOT = ROOT / "models"
OUTPUT_ROOT = ROOT / "output"
IMAGES_OUTPUT = OUTPUT_ROOT / "images"  # DIM expects output_folder/images/
COLMAP_DB = OUTPUT_ROOT / "database.db"
FORCE_RIG_FALLBACK = True

DATASET = DATA_ROOT / "S003"

CAM_FILES = {
    "cam01": str(DATASET / "cam01.mpeg"),
    "cam02": str(DATASET / "cam02.mpeg"),
    "cam03": str(DATASET / "cam03.mpeg"),
    "cam04": str(DATASET / "cam04.mpeg"),
}

# Depth Anything src mount (per constraint)
DEPTH_ANYTHING_SRC = str(MODELS_ROOT / "depth_anything_v3" / "src")

# SAM2 root
SAM2_ROOT = str(MODELS_ROOT / "sam2")

# DIM module name (installed or imported from models/deep-image-matching)
DIM_MODULE_NAME = "deep_image_matching"

# GPU allocation mapping cam -> cuda index
GPU_ALLOCATION = {
    "cam01": 0,
    "cam02": 1,
    "cam03": 2,
    "cam04": 3,
}

# SAM checkpoint (default suggestion)
# SAM2
SAM_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM_MODEL_CHECKPOINT = "models/sam2/checkpoints/sam2.1_hiera_large.pt"


# DA3 default model name
DA3_MODEL_NAME = "da3-large"

# Open3D
O3D_WINDOW_NAME = "UGRC: Dynamic 3D Reconstruction"

