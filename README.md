# Dynamic Multi-View 3D Reconstruction Using SAM2 and Depth Anything 3

### Undergraduate Research in Computer Science (UGRC), IIT Madras

**Author:** Tharaneeshwaran V U (CS25E053)  
**Guide:** Prof. Ayon Chakraborty

![System Architecture Diagram](/title_card.png)

## 1. Introduction

This repository contains the complete implementation of a dynamic 3D reconstruction pipeline developed as part of the Undergraduate Research in Computer Science (UGRC) program at IIT Madras.

The objective of this work is to explore how recent foundation models, specifically **SAM2** for video segmentation and **Depth Anything 3 (DA3)** for monocular depth can be combined to produce dense 3D point clouds from multiple camera views without relying on classical structure-from-motion (SfM) or multi-view stereo (MVS).

Traditional pipelines such as COLMAP depend heavily on accurate feature correspondences, precise camera calibration, and static scenes. They also incur significant computational cost. Dynamic scenes in particular pose a severe challenge. The approach in this repository avoids feature matching entirely and instead reconstructs 3D geometry by directly lifting predicted depth maps into 3D and aligning them across cameras. This makes the pipeline significantly simpler from an algorithmic perspective and applicable even when motion or occlusion breaks classical photogrammetry.

## 2. High-Level Description of the Pipeline

The system processes multi-view video sequences and produces temporally coherent 3D point clouds. The entire process can be executed using a single command via `main.py`. The workflow consists of five sequential stages, detailed in the table below.

| Stage | Technical Description |
| :--- | :--- |
| **1. Frame Extraction** | Each camera’s video stream is decomposed into per-frame RGB images. These are stored in a structured directory layout under `output/frames/<camID>/`. |
| **2. Camera Calibration** | When calibration data is unavailable or inconsistent—as is the case with the MultiScene360 dataset—the system uses a cylindrical-rig fallback calibration. This fallback provides approximate extrinsic parameters in a plausible multi-camera arrangement and enables multi-view fusion. |
| **3. Segmentation (SAM2)** | SAM2 is applied to each camera stream to produce consistent pixel-accurate masks of the target object across time. The system uses a single bounding-box guidance frame and propagates the segmentation through the video sequence. |
| **4. Depth Estimation (DA3)** | Depth Anything 3 generates monocular depth maps for every frame. The model is applied camera-wise and stored in a standardized format (`prediction.npz`) containing depth, intrinsics, and extrinsics. |
| **5. Multi-View Fusion** | Depth maps along with SAM2 masks are back-projected into 3D coordinates. The per-camera point clouds are transformed into a common world coordinate frame and merged. The pipeline optionally performs ICP refinement to improve geometric consistency across cameras. |

## 3. Repository Structure

The project allows for modular execution of specific pipeline stages. The directory structure is organized as follows:

```text
├── data/                   # Input video sequences and calibration files
├── output/                 # Generated artifacts
│   ├── frames/             # Extracted RGB frames per camera
│   ├── masks/              # SAM2 segmentation masks
│   ├── depth/              # DA3 depth maps (.npz format)
│   └── clouds/             # Final fused PLY point clouds
├── src/                    # Source code modules
│   ├── calibration.py      # Cylindrical rig fallback logic
│   ├── extraction.py       # Video frame decomposition
│   ├── sam_segment.py      # SAM2 integration and mask propagation
│   ├── depth_estim.py      # Depth Anything 3 inference
│   └── fusion.py           # Back-projection and ICP refinement
├── main.py                 # Entry point for the full pipeline
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
