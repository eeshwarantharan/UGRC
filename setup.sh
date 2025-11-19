#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="ugrc_env"
PYTHON_VERSION="3.10"
CONDA=$(which conda || true)

if [ -z "$CONDA" ]; then
  echo "conda not found. Install Miniconda/Anaconda and re-run."
  exit 1
fi

echo "Creating conda env: $ENV_NAME (python $PYTHON_VERSION)..."
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

echo "Activating $ENV_NAME..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Installing PyTorch for CUDA 12.4..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

echo "Installing pinned Python packages..."
pip install --upgrade pip
pip install open3d
pip install pycolmap  # optional; if pycolmap isn't available on platform, you can skip and DIM fallback will still produce db
pip install h5py
pip install matplotlib
pip install jupyterlab

mkdir -p models
cd models

echo "Cloning required repositories into models/ (if already present, skip/overwrite)..."
# Replace URLs if you host forks
git clone https://github.com/3DOM-FBK/deep-image-matching.git || echo "deep-image-matching exists"
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git depth_anything_v3 || echo "Depth-Anything-3 exists"
git clone https://github.com/facebookresearch/sam2.git sam2 || echo "sam2 exists"

cd ..

echo "Installing model repos in editable mode (this may compile SAM kernels)..."
# DIM
if [ -f "models/deep-image-matching/setup.py" ] || [ -f "models/deep-image-matching/pyproject.toml" ]; then
  pip install -e models/deep-image-matching
else
  echo "Warning: deep-image-matching repo not installable in-place. You can still import it by PYTHONPATH adjustments."
fi

# Depth Anything: we'll rely on src import per constraint. Optionally install if packaging exists.
if [ -f "models/depth_anything_v3/setup.py" ] || [ -f "models/depth_anything_v3/pyproject.toml" ]; then
  pip install -e models/depth_anything_v3 || echo "Editable install for Depth-Anything failed or already installed"
else
  echo "Depth-Anything-3 uses src import; we'll add its src path in runtime."
fi

# SAM2 requires editable install to compile CUDA extensions
if [ -f "models/sam2/setup.py" ] || [ -f "models/sam2/pyproject.toml" ]; then
  pip install -e models/sam2 || echo "Failed to pip install -e models/sam2; check nvcc/cuda"
else
  echo "SAM2 not found at models/sam2; ensure repo is cloned correctly."
fi

pip install numpy==1.26.4
pip install opencv-python-headless==4.9.0.80

echo ""
echo "Setup done. Activate with: conda activate $ENV_NAME"
echo "Notes:"
echo "- If SAM compilation fails, SAM still works without some optional CUDA post-processing."
echo "- DepthAnything3 will be imported by adding its src to sys.path (see src/config.py)."

