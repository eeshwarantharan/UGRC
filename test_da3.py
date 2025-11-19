import torch
from depth_anything_3.api import DepthAnything3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load nested-giant-large from HuggingFace
model = DepthAnything3.from_pretrained("depth-anything/da3nested-giant-large")
model = model.to(device=device)

# List of images (absolute or relative paths)
images = [
    "data/S003/frames/cam01/000178.jpg"
]

# Run full DA3 inference
prediction = model.inference(
    images,
    export_dir=None,           # don't export 3d files
    export_format="npz"         # no file output, just Python tensors
)

# Print outputs
print("DEPTH:", prediction.depth.shape)
print("DEPTH RANGE:", prediction.depth.min().item(), prediction.depth.max().item())
print("CONF:", prediction.conf.shape)
print("EXTRINSICS:", prediction.extrinsics.shape)
print("INTRINSICS:", prediction.intrinsics.shape)
