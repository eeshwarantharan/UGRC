import open3d as o3d
from pathlib import Path

def view_ply(path):
    pcd = o3d.io.read_point_cloud(str(path))
    o3d.visualization.draw_geometries([pcd])

