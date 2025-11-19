#!/usr/bin/env python3
"""
gui_viewer.py

12-panel synchronized playback + fused 3D view toggle.

Layout (Option D):
Top row:    RGB (cam01 cam02 cam03 cam04)
Middle row: Depth heatmaps (cam01..cam04)
Bottom row: Masks overlayed on RGB (cam01..cam04)

Controls: Play / Pause / Prev / Next / Slider / Record / Toggle fused view
Fused view: Open3D 3D scene (merged point cloud) + 4 small thumbnails at bottom

Requires: open3d with gui (tested with Open3D 0.16+), numpy, cv2
"""

import argparse
import os
import sys
import time
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np
import cv2

# Try to import Open3D GUI & rendering, otherwise fallback to a simpler loop that shows a composite image via cv2.imshow
try:
    import open3d as o3d
    from open3d import visualization
    from open3d.visualization import gui, rendering
    O3D_AVAILABLE = True
except Exception:
    O3D_AVAILABLE = False

# Defaults
DEFAULT_CAM_KEYS = ["cam01", "cam02", "cam03", "cam04"]
OUTPUT_FRAMES = Path("output/frames")
OUTPUT_DEPTH = Path("output/depth")
OUTPUT_MASKS = Path("output/masks")
OUTPUT_CLOUDS = Path("output/clouds")

# UI constants
THUMB_W = 360
THUMB_H = 202
DEPTH_CMAP = cv2.COLORMAP_INFERNO
MASK_COLOR = (0, 0, 255)  # BGR for overlay (we will convert to RGB array later)

# Utility functions ---------------------------------------------------------

def list_frames_for_cam(cam_key: str) -> List[Path]:
    d = OUTPUT_FRAMES / cam_key
    if not d.exists():
        return []
    imgs = sorted(list(d.glob("*.jpg")))
    return imgs

def load_rgb_frame(cam_key: str, frame_idx: int) -> Optional[np.ndarray]:
    imgs = list_frames_for_cam(cam_key)
    if not imgs:
        return None
    if frame_idx < 0 or frame_idx >= len(imgs):
        return None
    im = cv2.imread(str(imgs[frame_idx]))
    if im is None:
        return None
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def load_depth_for_cam(cam_key: str, frame_idx: int) -> Optional[np.ndarray]:
    # Look for prediction.npz (DA3 output) or depth.npy
    ddir = OUTPUT_DEPTH / cam_key
    if not ddir.exists():
        return None
    npz = ddir / "prediction.npz"
    if npz.exists():
        try:
            z = np.load(npz, allow_pickle=True)
            depth = z.get("depth", None)
            if depth is None:
                return None
            # depth may be [N,H,W] or [H,W]
            if depth.ndim == 3:
                if frame_idx < depth.shape[0]:
                    return np.asarray(depth[frame_idx])
                else:
                    return np.asarray(depth[0])
            elif depth.ndim == 2:
                return np.asarray(depth)
        except Exception:
            return None

    # fallback: depth.npy (maybe stacked)
    dnpy = ddir / "depth.npy"
    if dnpy.exists():
        try:
            depth = np.load(dnpy, allow_pickle=True)
            if depth is None:
                return None
            if depth.ndim == 3:
                if frame_idx < depth.shape[0]:
                    return np.asarray(depth[frame_idx])
                else:
                    return np.asarray(depth[0])
            elif depth.ndim == 2:
                return np.asarray(depth)
        except Exception:
            return None
    return None

def load_mask_for_cam(cam_key: str, frame_idx: int) -> Optional[np.ndarray]:
    p = OUTPUT_MASKS / cam_key / f"mask_{frame_idx:06d}.npy"
    if p.exists():
        try:
            m = np.load(p)
            # ensure binary 0/255, shape matches frame or will be resized later
            if m.dtype != np.uint8:
                m = (m > 0).astype(np.uint8) * 255
            else:
                m = (m > 0).astype(np.uint8) * 255
            return m
        except Exception:
            return None
    return None

def depth_to_colormap(depth: np.ndarray, normalize=True, clip_max=None) -> np.ndarray:
    if depth is None:
        return None
    d = np.array(depth, dtype=np.float32)
    # replace zeros with nan for visualization
    if normalize:
        valid = d > 0
        if clip_max is None:
            maxv = np.nanpercentile(d[valid], 95) if np.any(valid) else 1.0
        else:
            maxv = clip_max
        if maxv <= 0:
            maxv = d.max() if d.max() > 0 else 1.0
        d_vis = np.zeros_like(d)
        if np.any(valid):
            d_vis[valid] = (d[valid] - d[valid].min()) / max(1e-6, (maxv - d[valid].min()))
            d_vis = np.clip(d_vis, 0.0, 1.0)
        d_vis = (d_vis * 255).astype(np.uint8)
    else:
        d_vis = np.clip((d * 255.0).astype(np.uint8), 0, 255)
    cmap = cv2.applyColorMap(d_vis, DEPTH_CMAP)
    cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    return cmap

def overlay_mask_on_rgb(rgb: np.ndarray, mask: np.ndarray, alpha=0.5) -> np.ndarray:
    if rgb is None:
        return None
    h, w, _ = rgb.shape
    if mask is None:
        return rgb
    # resize mask if needed
    if mask.shape != (h, w):
        mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_r = mask
    mask_bool = mask_r > 0
    overlay = rgb.copy().astype(np.float32) / 255.0
    overlay[mask_bool] = (overlay[mask_bool] * (1.0 - alpha) + np.array(MASK_COLOR[::-1]) / 255.0 * alpha)
    return (overlay * 255).astype(np.uint8)

# Open3D helpers -------------------------------------------------------------

def load_merged_cloud_for_frame(frame_idx: int) -> Optional[o3d.geometry.PointCloud]:
    path = OUTPUT_CLOUDS / f"merged_frame_{frame_idx:06d}.ply"
    if not path.exists():
        return None
    try:
        pcd = o3d.io.read_point_cloud(str(path))
        return pcd
    except Exception:
        return None

# GUI application ------------------------------------------------------------

class UGRCViewerApp:
    def __init__(self, cams: List[str], start: int, end: int, frame_step=1):
        self.cams = cams
        self.start = start
        self.end = end
        self.step = frame_step
        self.frame_count = max(0, (end - start) // frame_step + 1)
        self.current_frame_idx = 0  # index into range: frame_number = start + idx*step
        self.playing = False
        self.recording = False
        self.record_dir = None
        self._load_frame_lists()

        # Open3D GUI window / widgets
        if O3D_AVAILABLE:
            self._init_o3d_gui()
        else:
            print("[WARN] Open3D GUI not available. Use fallback image loop.")
            # fallback variables for cv2 loop
            self.window_name = "UGRC Viewer (fallback)"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Preload small cache for speed
        self.cache = {}

    def _load_frame_lists(self):
        # find per-camera frame lists; we will use index relative to start
        self.cam_frames = {}
        self.n_frames_per_cam = {}
        for cam in self.cams:
            imgs = list_frames_for_cam(cam)
            self.cam_frames[cam] = imgs
            self.n_frames_per_cam[cam] = len(imgs)
        # compute max frames available (min across cams is fine)
        self.available_frames = min((len(v) for v in self.cam_frames.values()), default=0)
        # clamp end to available if possible
        if self.available_frames > 0:
            max_idx = min(self.end, self.start + self.available_frames - 1)
            if max_idx < self.end:
                print(f"[INFO] Clamping end to {max_idx} due to available frames.")
                self.end = max_idx
                self.frame_count = max(0, (self.end - self.start) // self.step + 1)

    # ----------------- Open3D GUI initialization -----------------
    def _init_o3d_gui(self):
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("UGRC Viewer", 1600, 1000)

        # root layout: two columns: left 3D view, right 12-panel grid (we will create a grid)
        self._create_controls()
        self._create_grid_panels()
        self._create_3d_scene()

        # Start timer for playback (interval in ms)
        self.timer_msec = 100  # 10 FPS default (you can change)
        gui.Application.instance.set_on_timer(self.timer_msec, self._on_timer)

    def _create_controls(self):
        # Top control bar
        em = self.window.theme.font_size
        ctrl = gui.Vert(0.5 * em)
        ctrl.style = {"margin": 8}
        # Play / Pause / Prev / Next
        row = gui.Horiz(0.5 * em)
        self.btn_prev = gui.Button("◀ Prev")
        self.btn_prev.set_on_clicked(lambda: self._on_prev())
        self.btn_play = gui.Button("Play ▶")
        self.btn_play.set_on_clicked(lambda: self._on_playpause())
        self.btn_next = gui.Button("Next ▶")
        self.btn_next.set_on_clicked(lambda: self._on_next())
        self.btn_record = gui.Button("Record ⏺")
        self.btn_record.set_on_clicked(lambda: self._on_record_toggle())
        self.btn_toggle_view = gui.Button("Show Fused View")
        self.btn_toggle_view.set_on_clicked(lambda: self._on_toggle_view())
        row.add_child(self.btn_prev)
        row.add_child(self.btn_play)
        row.add_child(self.btn_next)
        row.add_child(self.btn_record)
        row.add_child(self.btn_toggle_view)

        # Slider
        self.slider = gui.Slider(gui.Slider.INT)
        self.slider.set_limits(0, max(0, self.frame_count - 1))
        self.slider.set_on_value_changed(lambda v: self._on_slider(int(v)))
        ctrl.add_child(row)
        ctrl.add_child(self.slider)

        # status label
        self.lbl_status = gui.Label("Frame: 0 / 0")
        ctrl.add_child(self.lbl_status)

        # layout placement
        self.window.add_child(ctrl)
        # place later with grid, keep reference
        self.control_widget = ctrl

    def _create_grid_panels(self):
        # Create a 3x4 grid of Image widgets
        self.panel_grid = gui.Grid(3, 4)
        # store image widgets for 12 panels in order: row0 RGB x4, row1 depth x4, row2 mask x4
        self.img_widgets = []
        for r in range(3):
            for c in range(4):
                w = gui.ImageWidget()
                w.scale_to_fit = True
                # set some preferred size
                w.frame.set_size_policy(gui.Widget.SizePolicy.EXPANDING, gui.Widget.SizePolicy.EXPANDING)
                self.panel_grid.add_child(w, r, c)
                self.img_widgets.append(w)

        # We'll place the controls at top-left and the grid below them on the right column
        # Make a two-column layout: left = 3D scene (placeholder), right = controls + panels
        # Create a container for right column
        right_col = gui.Vert(0.5 * self.window.theme.font_size)
        right_col.add_child(self.control_widget)
        right_col.add_child(self.panel_grid)
        # Save for swapping when fused view toggled
        self.right_col = right_col

        # main layout: HBox: [3DViewPlaceholder | right_col]
        self.main_h = gui.Horiz(0.5 * self.window.theme.font_size)
        # left placeholder for 3D scene (we will create actual scene widget later)
        self.left_container = gui.Vert(0.5 * self.window.theme.font_size)
        self.left_container.style = {"margin": 6}
        self.main_h.add_child(self.left_container)
        self.main_h.add_child(self.right_col)

        # finally add main_h to window's root (below the control bar already added)
        self.window.add_child(self.main_h)

    def _create_3d_scene(self):
        # Create an Open3D SceneWidget for rendering the merged pointcloud
        self.scene_widget = gui.SceneWidget()
        self.scene = self.scene_widget.scene
        # add lighting
        self.scene.set_background([0, 0, 0, 1.0])
        self.scene.scene.set_sun_light([0.3, -1, -0.3], [1.0, 1.0, 1.0], 100000)
        self.left_container.add_child(self.scene_widget)
        # add camera thumbnail placeholders (we will add 4 thumbnails under the 3D view)
        thumbs = gui.Horiz(0.5 * self.window.theme.font_size)
        self.thumb_widgets = []
        for i in range(4):
            imw = gui.ImageWidget()
            imw.scale_to_fit = True
            thumbs.add_child(imw)
            self.thumb_widgets.append(imw)
        self.left_container.add_child(thumbs)

        # initially show grid (12-panel) mode; fused view will replace right_col with scene+thumbs full-screen
        self.fused_mode = False
        self._clear_scene()

    def _clear_scene(self):
        # remove all geometry
        try:
            for key in list(self.scene.get_geometry_names()):
                self.scene.remove_geometry(key)
        except Exception:
            pass

    # ----------------- GUI event handlers -----------------
    def _on_playpause(self):
        self.playing = not self.playing
        self.btn_play.text = "Pause ⏸" if self.playing else "Play ▶"

    def _on_prev(self):
        self.playing = False
        self.btn_play.text = "Play ▶"
        self.current_frame_idx = max(0, self.current_frame_idx - 1)
        self._render_current_frame()

    def _on_next(self):
        self.playing = False
        self.btn_play.text = "Play ▶"
        self.current_frame_idx = min(self.frame_count - 1, self.current_frame_idx + 1)
        self._render_current_frame()

    def _on_slider(self, v: int):
        self.playing = False
        self.btn_play.text = "Play ▶"
        self.current_frame_idx = v
        self._render_current_frame()

    def _on_record_toggle(self):
        if not self.recording:
            # start recording
            outdir = Path("output") / "recordings" / time.strftime("%Y%m%d_%H%M%S")
            outdir.mkdir(parents=True, exist_ok=True)
            self.record_dir = outdir
            self.recording = True
            self.btn_record.text = "Stop ⏹"
            print(f"[REC] Recording frames to {outdir}")
        else:
            # stop recording
            print(f"[REC] Stopped recording. Files saved to {self.record_dir}")
            self.recording = False
            self.btn_record.text = "Record ⏺"

    def _on_toggle_view(self):
        # Switch between 12-panel grid and fused 3D view
        self.fused_mode = not self.fused_mode
        if self.fused_mode:
            self.btn_toggle_view.text = "Show 12-panel"
            # make the right column replaced by minimal controls when fused
            # expand left container to contain scene full height — already left has scene
            # keep right column small (thumbnails + slider)
            # nothing fancy — just keep existing layout but hide panel_grid
            self.panel_grid.visible = False
        else:
            self.btn_toggle_view.text = "Show Fused View"
            self.panel_grid.visible = True
        # re-render to update view
        self._render_current_frame()

    def _on_timer(self):
        if not self.playing:
            return True
        self.current_frame_idx += 1
        if self.current_frame_idx >= self.frame_count:
            self.current_frame_idx = 0
        self.slider.set_value(self.current_frame_idx)
        self._render_current_frame()
        return True

    # ----------------- Rendering per frame -----------------
    def _frame_number(self) -> int:
        return self.start + self.current_frame_idx * self.step

    def _render_current_frame(self):
        frame_num = self._frame_number()
        self.lbl_status.text = f"Frame: {frame_num}  (idx {self.current_frame_idx} / {self.frame_count - 1})"

        # load per-camera images, depth, masks
        rgbs = []
        depths = []
        masks = []
        for cam in self.cams:
            rgb = load_rgb_frame(cam, frame_num - self.start)
            depth = load_depth_for_cam(cam, frame_num - self.start)
            mask = load_mask_for_cam(cam, frame_num)
            rgbs.append(rgb)
            depths.append(depth)
            masks.append(mask)

        # prepare panels
        # Top row: RGB thumbnails
        for i in range(4):
            img = rgbs[i]
            if img is None:
                arr = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
                cv2.putText(arr, f"No RGB ({self.cams[i]})", (10, THUMB_H//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                img_disp = arr
            else:
                img_disp = cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
            if O3D_AVAILABLE:
                self.img_widgets[i].update_image(o3d.geometry.Image(img_disp))
            else:
                rgbs[i] = img_disp

        # Middle row: Depth color maps
        for i in range(4):
            d = depths[i]
            if d is None:
                arr = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
                cv2.putText(arr, f"No Depth ({self.cams[i]})", (10, THUMB_H//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                img_disp = arr
            else:
                cm = depth_to_colormap(d)
                img_disp = cv2.resize(cm, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
            if O3D_AVAILABLE:
                self.img_widgets[4 + i].update_image(o3d.geometry.Image(img_disp))
            else:
                depths[i] = img_disp

        # Bottom row: Mask overlays
        for i in range(4):
            rgb = rgbs[i] if rgbs[i] is not None else None
            m = masks[i]
            if rgb is None:
                arr = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
                cv2.putText(arr, f"No RGB ({self.cams[i]})", (10, THUMB_H//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                img_disp = arr
            else:
                # rgb currently may be resized already to THUMB dimension if fallback; if not, resize
                if rgb.shape[0] != THUMB_H or rgb.shape[1] != THUMB_W:
                    rgb_res = cv2.resize(rgb, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
                else:
                    rgb_res = rgb
                img_disp = overlay_mask_on_rgb(rgb_res, m)
            if O3D_AVAILABLE:
                self.img_widgets[8 + i].update_image(o3d.geometry.Image(img_disp))
            else:
                masks[i] = img_disp

        # In fused mode, render merged cloud in the 3D scene
        if self.fused_mode and O3D_AVAILABLE:
            # load merged cloud
            pcd = load_merged_cloud_for_frame(frame_num)
            self._clear_scene()
            if pcd is not None and len(pcd.points) > 0:
                mat = rendering.MaterialRecord()
                mat.shader = "defaultUnlit"
                mat.point_size = 2.0
                self.scene.add_geometry("merged", pcd, mat)
                # center camera on cloud
                bbox = pcd.get_axis_aligned_bounding_box()
                center = bbox.get_center()
                self.scene_widget.setup_camera(60.0, bbox, center)
            else:
                # nothing to show: display message in scene by overlaying a small image in left container
                pass

            # update the 4 thumbnails under the 3D view
            for i in range(4):
                thumb_img = rgbs[i] if isinstance(rgbs[i], np.ndarray) else np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
                self.thumb_widgets[i].update_image(o3d.geometry.Image(thumb_img))

        # If recording, dump the assembled 12-panel image for video later
        if self.recording:
            out_img = self._compose_12panel_image(rgbs, depths, masks)
            if out_img is not None and self.record_dir is not None:
                fname = self.record_dir / f"frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(fname), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    def _compose_12panel_image(self, rgbs, depths_cmaps, masks_over):
        # Compose a single large RGB image with 3 rows x 4 cols
        rows = []
        for row_idx in range(3):
            cols = []
            for col_idx in range(4):
                idx = row_idx * 4 + col_idx
                if row_idx == 0:
                    im = rgbs[col_idx]
                    if im is None:
                        im = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
                    else:
                        im = cv2.resize(im, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
                elif row_idx == 1:
                    d = depths_cmaps[col_idx]
                    if d is None:
                        im = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
                    else:
                        im = cv2.resize(d, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
                else:
                    msk = masks_over[col_idx]
                    rgb = rgbs[col_idx]
                    if rgb is None:
                        rgb = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
                    else:
                        rgb = cv2.resize(rgb, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
                    im = overlay_mask_on_rgb(rgb, msk)
                cols.append(im)
            rows.append(np.hstack(cols))
        full = np.vstack(rows)
        return full

    # ----------------- Fallback (cv2) loop if O3D GUI not available -----------------
    def run_fallback_loop(self):
        # show composed 12-panel with keyboard control
        self.current_frame_idx = 0
        while True:
            frame_num = self._frame_number()
            # load and compose
            rgbs = []
            depths = []
            masks = []
            for cam in self.cams:
                rgbs.append(load_rgb_frame(cam, frame_num - self.start))
                depths.append(depth_to_colormap(load_depth_for_cam(cam, frame_num - self.start)) if load_depth_for_cam(cam, frame_num - self.start) is not None else None)
                masks.append(load_mask_for_cam(cam, frame_num))
            out = self._compose_12panel_image(rgbs, depths, masks)
            if out is None:
                out = np.zeros((THUMB_H * 3, THUMB_W * 4, 3), dtype=np.uint8)
            cv2.imshow(self.window_name, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            k = cv2.waitKey(100 if self.playing else 0)
            if k == ord("q"):
                break
            elif k == ord(" "):
                self._on_playpause()
            elif k == ord("n"):
                self._on_next()
            elif k == ord("p"):
                self._on_prev()
            elif k == ord("r"):
                self._on_record_toggle()
            if self.playing:
                self.current_frame_idx = (self.current_frame_idx + 1) % max(1, self.frame_count)
            # handle recording
            if self.recording and self.record_dir:
                cv2.imwrite(str(self.record_dir / f"frame_{frame_num:06d}.jpg"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

        cv2.destroyAllWindows()

    # ----------------- Start app -----------------
    def run(self):
        if O3D_AVAILABLE:
            self._render_current_frame()
            gui.Application.instance.run()
        else:
            self.run_fallback_loop()


# CLI ------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=1000)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--cams", nargs="+", default=DEFAULT_CAM_KEYS)
    return p.parse_args()

def main():
    args = parse_args()
    app = UGRCViewerApp(cams=args.cams, start=args.start, end=args.end, frame_step=args.step)
    app.run()

if __name__ == "__main__":
    main()

