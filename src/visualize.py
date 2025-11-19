# src/visualize.py
"""
Small Open3D-based visualization helpers:
 - view_pointcloud(path_or_o3d)
 - save_snapshot(path, vis_filename)
"""
import open3d as o3d
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger("visualize")
logging.basicConfig(level=logging.INFO)

def view_pointcloud(ply_path_or_pcd):
    """
    Live interactive viewer; blocks until closed.
    """
    if isinstance(ply_path_or_pcd, str) or isinstance(ply_path_or_pcd, Path):
        pcd = o3d.io.read_point_cloud(str(ply_path_or_pcd))
    elif isinstance(ply_path_or_pcd, o3d.geometry.PointCloud):
        pcd = ply_path_or_pcd
    else:
        raise ValueError("Unsupported input to view_pointcloud")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Merged PointCloud", width=1280, height=720)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)
    vis.run()
    vis.destroy_window()

def save_png_snapshot(ply_path, out_png):
    pcd = o3d.io.read_point_cloud(str(ply_path))
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)
    vis.add_geometry(pcd)
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(str(out_png))
    vis.destroy_window()
    logger.info("Saved snapshot %s", out_png)

