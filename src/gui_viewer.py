"""
gui_viewer.py — Open3D-based multi-panel GUI for UGRC.

Controls:
 - Space: play/pause
 - Left/Right: step frames
 - R: record current layout as frames
"""

import os
import sys
from pathlib import Path
import time
import threading
import logging
import numpy as np
import cv2

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

logger = logging.getLogger("gui_viewer")
logging.basicConfig(level=logging.INFO)

# Paths must match main.py / pipeline outputs
OUTPUT_FRAMES = Path("output/frames")
OUTPUT_DEPTH = Path("output/depth")
OUTPUT_MASKS = Path("output/masks")
OUTPUT_CLOUDS = Path("output/clouds")

# GUI constants
WINDOW_TITLE = "UGRC — Multi-Cam Viewer"
WIN_WIDTH = 1400
WIN_HEIGHT = 900

# UI helper to convert np image -> o3d texture
def to_o3d_image(rgb):
    if rgb is None:
        return None
    if rgb.dtype == np.uint8:
        return o3d.geometry.Image(rgb)
    # convert float [0,1] -> uint8
    im = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return o3d.geometry.Image(im)

# main GUI class
class MultiCamViewer:
    def __init__(self, start=0, end=10, cam_keys=None):
        if cam_keys is None:
            cam_keys = ["cam01", "cam02", "cam03", "cam04"]
        self.cam_keys = cam_keys
        self.start = start
        self.end = end
        self.index = start
        self.is_playing = False
        self.play_thread = None
        self.play_delay = 0.1  # seconds
        self.recording = False
        self.record_out = Path("output/gui_record")
        self.record_out.mkdir(parents=True, exist_ok=True)

        # Open3D app & window
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(WINDOW_TITLE, WIN_WIDTH, WIN_HEIGHT)
        self.window.set_on_key(self._on_key)
        self._build_layout()

        # initial load
        self._refresh_all_views()

    def _build_layout(self):
        em = self.window.theme.font_size
        # layout: top (RGB row), mid (Depth row), bottom (Masks row) => 3 rows x 4 columns = 12 panels
        grid = gui.Vert(0)
        top = gui.Horiz(0)
        mid = gui.Horiz(0)
        bot = gui.Horiz(0)

        # For each cam create image widget containers
        self.rgb_widgets = []
        self.depth_widgets = []
        self.mask_widgets = []

        for cam in self.cam_keys:
            # rgb
            iv = gui.ImageWidget()
            iv.set_preferred_size(320, 180)
            self.rgb_widgets.append(iv)
            top.add_child(iv)

        for cam in self.cam_keys:
            iv = gui.ImageWidget()
            iv.set_preferred_size(320, 180)
            self.depth_widgets.append(iv)
            mid.add_child(iv)

        for cam in self.cam_keys:
            iv = gui.ImageWidget()
            iv.set_preferred_size(320, 180)
            self.mask_widgets.append(iv)
            bot.add_child(iv)

        grid.add_child(top)
        grid.add_child(gui.Label(""))  # spacer
        grid.add_child(mid)
        grid.add_child(gui.Label(""))  # spacer
        grid.add_child(bot)

        # Controls at right side
        ctrl = gui.Vert(0)
        # frame slider
        self.slider = gui.Slider(gui.Slider.INT)
        self.slider.set_limits(self.start, self.end)
        self.slider.int_value = self.index
        self.slider.set_on_value_changed(self._on_slider_changed)
        ctrl.add_child(gui.Label("Frame"))
        ctrl.add_child(self.slider)
        # play/pause
        self.play_btn = gui.Button("Play")
        self.play_btn.set_on_clicked(self._on_play_clicked)
        ctrl.add_child(self.play_btn)
        # step buttons
        h = gui.Horiz(0)
        btn_prev = gui.Button("◀ Prev")
        btn_next = gui.Button("Next ▶")
        btn_prev.set_on_clicked(lambda: self._step(-1))
        btn_next.set_on_clicked(lambda: self._step(1))
        h.add_child(btn_prev)
        h.add_child(btn_next)
        ctrl.add_child(h)
        # fused toggle
        self.fused_toggle = gui.Toggle("Fused 3D View")
        self.fused_toggle.set_on_clicked(self._on_fused_toggle)
        ctrl.add_child(self.fused_toggle)
        # record
        self.rec_btn = gui.Button("Record")
        self.rec_btn.set_on_clicked(self._on_record)
        ctrl.add_child(self.rec_btn)
        # status
        self.status = gui.Label("Ready")
        ctrl.add_child(self.status)

        # Add grid + ctrl to window layout
        main = gui.Horiz(0)
        main.add_child(grid)
        main.add_child(ctrl)
        self.window.add_child(main)

        # 3D scene (for fused) - we'll overlay on top when toggled
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.visible = False
        self.window.add_child(self.scene)

    # keyboard handler
    def _on_key(self, wnd, event):
        if event.type == gui.KeyEvent.DOWN:
            if event.key == gui.KeyName.SPACE:
                self._on_play_clicked()
                return True
            if event.key == gui.KeyName.RIGHT:
                self._step(1); return True
            if event.key == gui.KeyName.LEFT:
                self._step(-1); return True
            if event.key == "R":
                self._on_record(); return True
        return False

    def _on_slider_changed(self, value):
        self.index = int(value)
        self._refresh_all_views()

    def _on_play_clicked(self):
        if self.is_playing:
            self.is_playing = False
            self.play_btn.text = "Play"
        else:
            self.is_playing = True
            self.play_btn.text = "Pause"
            self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self.play_thread.start()

    def _step(self, delta):
        self.index = max(self.start, min(self.end, self.index + delta))
        self.slider.int_value = self.index
        self._refresh_all_views()

    def _on_fused_toggle(self):
        self.scene.visible = self.fused_toggle.is_checked()
        # When toggled on, load fused cloud for current frame
        if self.scene.visible:
            self._load_fused_cloud(self.index)

    def _on_record(self):
        self.recording = not self.recording
        self.rec_btn.text = "Stop" if self.recording else "Record"
        if self.recording:
            self.status.text = f"Recording → {self.record_out}"
        else:
            self.status.text = "Stopped recording"

    def _play_loop(self):
        while self.is_playing:
            t0 = time.time()
            self._step(1)
            if self.recording:
                self._save_current_layout()
            dt = time.time() - t0
            time.sleep(max(0.01, self.play_delay - dt))

    # UI refresh: load images from disk for each cam, update widgets
    def _refresh_all_views(self):
        # for each cam update RGB, depth, mask
        for idx, cam in enumerate(self.cam_keys):
            # rgb
            rgb_path = OUTPUT_FRAMES / cam / f"{self.index:06d}.jpg"
            if rgb_path.exists():
                rgb = cv2.imread(str(rgb_path))[:,:,::-1]  # BGR->RGB
                rgb_o3d = to_o3d_image(rgb)
                w = self.rgb_widgets[idx]
                if rgb_o3d is not None:
                    w.update_image(rgb_o3d)
            else:
                # show placeholder
                w = self.rgb_widgets[idx]
                blank = 255 * np.ones((180, 320, 3), dtype=np.uint8)
                blank[:,:] = (50,50,50)
                w.update_image(to_o3d_image(blank))

            # depth
            depth_np = None
            ddir = OUTPUT_DEPTH / cam
            # prefer per-frame prediction_{frame}.npz
            npz_f = ddir / f"prediction_{self.index:06d}.npz"
            if npz_f.exists():
                try:
                    dd = np.load(npz_f)
                    depth_np = dd.get("depth")
                    if depth_np is not None:
                        # depth shape can be (1,H,W) or (H,W)
                        if depth_np.ndim == 3:
                            depth_np = depth_np[0]
                except Exception:
                    depth_np = None
            else:
                # maybe a single prediction.npz saved — try that and pick frame 0
                alt = ddir / "prediction.npz"
                if alt.exists():
                    try:
                        dd = np.load(alt)
                        depth_np = dd.get("depth")
                        if depth_np is not None:
                            if depth_np.ndim == 3:
                                # Attempt to pick matching index; if out of range use 0
                                fid = min(self.index, depth_np.shape[0]-1)
                                depth_np = depth_np[fid]
                    except Exception:
                        depth_np = None

            if depth_np is not None:
                # Normalize to 0..255 for display
                dmin, dmax = float(np.nanmin(depth_np)), float(np.nanmax(depth_np))
                if dmax - dmin < 1e-6:
                    norm = np.zeros_like(depth_np, dtype=np.uint8)
                else:
                    norm = ((depth_np - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)
                # create colored map (jet)
                col = cv2.applyColorMap(norm, cv2.COLORMAP_JET)[:,:,::-1]
                self.depth_widgets[idx].update_image(to_o3d_image(col))
            else:
                blank = 255 * np.ones((180,320,3), dtype=np.uint8)
                blank[:,:] = (30,30,30)
                self.depth_widgets[idx].update_image(to_o3d_image(blank))

            # mask
            mask_p = OUTPUT_MASKS / cam / f"mask_{self.index:06d}.npy"
            if mask_p.exists():
                m = np.load(mask_p)
                # ensure HxW -> resize to widget size
                try:
                    mh, mw = m.shape
                    display = (m > 0).astype(np.uint8) * 255
                    display = cv2.resize(display, (320,180), interpolation=cv2.INTER_NEAREST)
                    display = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)
                    self.mask_widgets[idx].update_image(to_o3d_image(display))
                except Exception:
                    blank = 255 * np.ones((180,320,3), dtype=np.uint8)
                    blank[:,:] = (20,80,20)
                    self.mask_widgets[idx].update_image(to_o3d_image(blank))
            else:
                blank = 255 * np.ones((180,320,3), dtype=np.uint8)
                blank[:,:] = (30,30,30)
                self.mask_widgets[idx].update_image(to_o3d_image(blank))

        # if fused view visible -> update cloud
        if self.scene.visible:
            self._load_fused_cloud(self.index)

        self.status.text = f"Frame {self.index}"

    def _load_fused_cloud(self, frame_idx):
        # Look for merged PLY named merged_000{frame}.ply or merged_frame_{frame}.ply
        p1 = OUTPUT_CLOUDS / f"merged_{frame_idx:06d}.ply"
        p2 = OUTPUT_CLOUDS / f"merged_frame_{frame_idx:06d}.ply"
        target = p1 if p1.exists() else (p2 if p2.exists() else None)
        if target is None:
            self.scene.scene.clear_geometry()
            self.status.text = f"No fused cloud for {frame_idx}"
            return
        # load
        pcd = o3d.io.read_point_cloud(str(target))
        self.scene.scene.clear_geometry()
        self.scene.scene.add_geometry("fused", pcd, rendering.MaterialRecord())
        # set up camera
        bounds = pcd.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        extent = max(bounds.extent)
        cam = rendering.Camera()
        # create simple camera that frames bounding box
        self.scene.setup_camera(60.0, bounds, center)
        self.status.text = f"Loaded fused cloud: {target.name} (pts={np.asarray(pcd.points).shape[0]})"

    def _save_current_layout(self):
        # save each widget as png snapshot (quick approach: grab images from disk & store)
        out = self.record_out / f"{self.index:06d}"
        out.mkdir(parents=True, exist_ok=True)
        # save rgb widgets original images (from disk)
        for idx, cam in enumerate(self.cam_keys):
            rgb_path = OUTPUT_FRAMES / cam / f"{self.index:06d}.jpg"
            if rgb_path.exists():
                dst = out / f"{cam}_rgb.jpg"
                shutil.copy(rgb_path, dst)
            # depth & mask: capture what we displayed by exporting the widget image if possible
        logger.info("Saved record frame: %s", out)

# run entrypoint
def run_gui(start=0, end=10, cam_keys=None):
    viewer = MultiCamViewer(start=start, end=end, cam_keys=cam_keys)
    gui.Application.instance.run()

# CLI support
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=10)
    p.add_argument("--cams", nargs="+", default=None)
    a = p.parse_args()
    cams = a.cams if a.cams else ["cam01","cam02","cam03","cam04"]
    run_gui(start=a.start, end=a.end, cam_keys=cams)

