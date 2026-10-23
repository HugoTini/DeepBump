"""Pick an image, turn it into a heightmap, and export it.

Runs DeepBump color → normals, then normals → height. Launch from this folder:

    python gui.py
"""

import os
import queue
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import imageio.v3 as iio
import numpy as np
from numba import njit
from PIL import Image, ImageTk

import module_color_to_normals
import module_normals_to_height

PREVIEW_MAX = 480


def load_color_image(path):
    """Return C,H,W float32 RGB in [0, 1]."""
    img = iio.imread(path)
    if img.ndim == 2:
        img = img[:, :, None]
    img = np.asarray(img)
    if img.shape[-1] > 3:
        img = img[:, :, :3]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)
        peak = float(np.max(img)) if img.size else 1.0
        if peak > 1.0:
            img = img / peak
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=2)
    return np.transpose(img, (2, 0, 1))


def height_to_u8(height_chw):
    channel = np.clip(height_chw[0], 0.0, 1.0)
    return (channel * 255.0).astype(np.uint8)


def _bilinear(image, ys, xs):
    """Sample image (H, W) or (H, W, C) at row coords ys and column coords xs."""
    height, width = image.shape[:2]
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, height - 1)
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    wy = (ys - np.floor(ys))[:, None]
    wx = (xs - np.floor(xs))[None, :]
    if image.ndim == 3:
        wy = wy[..., None]
        wx = wx[..., None]
    v00 = image[y0][:, x0]
    v01 = image[y0][:, x1]
    v10 = image[y1][:, x0]
    v11 = image[y1][:, x1]
    return v00 * (1.0 - wy) * (1.0 - wx) + v01 * (1.0 - wy) * wx + v10 * wy * (1.0 - wx) + v11 * wy * wx


def _box_blur(image, radius):
    """Mean filter. Edges are reflected so the blur does not invent a border slope."""
    radius = int(radius)
    if radius < 1:
        return image
    padded = np.pad(image, radius, mode="reflect")
    height, width = padded.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    ys = np.arange(radius, height - radius)
    xs = np.arange(radius, width - radius)
    y0 = ys - radius
    y1 = ys + radius + 1
    x0 = xs - radius
    x1 = xs + radius + 1
    total = integral[y1][:, x1] - integral[y0][:, x1] - integral[y1][:, x0] + integral[y0][:, x0]
    area = float((radius * 2 + 1) ** 2)
    return total / area


def highpass_height(height_chw, amount):
    """High-pass the height image. amount is 0–100. Result stays in 0–1."""
    if amount <= 0:
        return height_chw
    src = np.asarray(height_chw[0], dtype=np.float64)
    short = min(src.shape)
    if short < 3:
        return height_chw
    gentle = max(2, int(round(short * 0.08)))
    mix = amount / 100.0
    radius = int(round(gentle * (1.0 - mix) + 2.0 * mix))
    radius = min(max(radius, 1), short - 1)
    low = src
    for _ in range(3):
        low = _box_blur(low, radius)
    detail = src - low
    peak = float(np.percentile(np.abs(detail), 99.5))
    if peak < 1e-8:
        centered = np.full(src.shape, 0.5, dtype=np.float32)
    else:
        # tanh keeps every pixel in range, so peaks fade instead of clipping into flats.
        centered = (0.5 + 0.5 * np.tanh(detail / peak)).astype(np.float32)
    out = np.empty(height_chw.shape, dtype=np.float32)
    out[0] = centered
    if out.shape[0] > 1:
        out[1:] = out[0]
    return out


def _focal(size):
    return 1.7 * min(size) * 0.5


def _visible_pixel_span(yaw, pitch, distance, height_scale, size, image_shape):
    """Pixel rectangle around the look-at point that fills the current view.

    Zooming in shrinks this rectangle, so the same vertex budget covers real pixels.
    """
    image_h, image_w = image_shape
    aspect = image_w / max(image_h, 1)
    focal = _focal(size)
    # The orbit camera looks at the origin. A screen pixel there covers this much world.
    world_per_pixel = max(distance, 1e-3) / focal
    margin = 1.2 + 0.35 * height_scale
    half_x = size[0] * 0.5 * world_per_pixel * margin
    half_z = size[1] * 0.5 * world_per_pixel * margin
    u0 = float(np.clip(0.5 - half_x / aspect, 0.0, 1.0))
    u1 = float(np.clip(0.5 + half_x / aspect, 0.0, 1.0))
    v0 = float(np.clip(0.5 - half_z, 0.0, 1.0))
    v1 = float(np.clip(0.5 + half_z, 0.0, 1.0))
    x0 = u0 * (image_w - 1)
    x1 = u1 * (image_w - 1)
    y0 = v0 * (image_h - 1)
    y1 = v1 * (image_h - 1)
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return 0.0, float(image_w - 1), 0.0, float(image_h - 1)
    return x0, x1, y0, y1


def _sample_grid(color_chw, height_chw, x0, x1, y0, y1, cols, rows):
    """Sample the pixel span at the requested vertex counts."""
    xs = np.linspace(x0, x1, cols)
    ys = np.linspace(y0, y1, rows)
    relief = _bilinear(height_chw[0], ys, xs)
    rgb = _bilinear(np.transpose(color_chw, (1, 2, 0)), ys, xs)
    return relief, np.clip(rgb, 0.0, 1.0), xs, ys


def _grid_shape(x0, x1, y0, y1, image_shape, distance, size, target_grid):
    """Vertex counts for this zoom.

    A settled view (target_grid <= 0) uses one vertex per heightmap pixel in the
    patch on screen, and stops there: closer zoom cannot invent more data.
    While dragging, target_grid is the long-side budget.
    """
    image_h, image_w = image_shape
    span_x = max(x1 - x0, 1.0)
    span_y = max(y1 - y0, 1.0)
    if target_grid is not None and target_grid > 0:
        scale = min(1.0, target_grid / max(span_x, span_y))
        cols = max(2, int(round(span_x * scale)) + 1)
        rows = max(2, int(round(span_y * scale)) + 1)
        return cols, rows

    aspect = image_w / max(image_h, 1)
    world_per_pixel = max(distance, 1e-3) / _focal(size)
    world_w = span_x / max(image_w - 1, 1) * aspect
    world_h = span_y / max(image_h - 1, 1)
    # Two vertices per screen pixel until the heightmap itself runs out of pixels.
    screen_x = world_w / world_per_pixel * 2.0
    screen_y = world_h / world_per_pixel * 2.0
    cols = max(2, int(round(min(span_x, screen_x))) + 1)
    rows = max(2, int(round(min(span_y, screen_y))) + 1)
    long_side = max(cols, rows)
    if long_side > 2048:
        shrink = 2048 / long_side
        cols = max(2, int(cols * shrink))
        rows = max(2, int(rows * shrink))
    return cols, rows


def render_relief(color_chw, height_chw, yaw, pitch, distance, height_scale, target_grid, size):
    """Draw the source image displaced by the heightmap. Returns a PIL image."""
    image_h, image_w = height_chw.shape[1], height_chw.shape[2]
    x0, x1, y0, y1 = _visible_pixel_span(
        yaw, pitch, distance, height_scale, size, (image_h, image_w)
    )
    cols, rows = _grid_shape(x0, x1, y0, y1, (image_h, image_w), distance, size, target_grid)
    relief, rgb, xs, ys = _sample_grid(color_chw, height_chw, x0, x1, y0, y1, cols, rows)
    rows, cols = relief.shape
    aspect = image_w / image_h
    col_u, row_v = np.meshgrid((xs / max(image_w - 1, 1) - 0.5) * aspect, ys / max(image_h - 1, 1) - 0.5)
    points = np.stack(
        [col_u, (relief - 0.5) * height_scale, row_v],
        axis=-1,
    ).astype(np.float64)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rotated = np.empty_like(points)
    rotated[..., 0] = points[..., 0] * cy + points[..., 2] * sy
    rotated[..., 1] = points[..., 1]
    rotated[..., 2] = -points[..., 0] * sy + points[..., 2] * cy
    pitched = np.empty_like(rotated)
    pitched[..., 0] = rotated[..., 0]
    pitched[..., 1] = rotated[..., 1] * cp - rotated[..., 2] * sp
    pitched[..., 2] = rotated[..., 1] * sp + rotated[..., 2] * cp
    view_z = np.maximum(pitched[..., 2] + distance, 0.05)

    light = np.array([0.35, 0.85, 0.4], dtype=np.float64)
    light /= np.linalg.norm(light)
    p00, p10, p01 = points[:-1, :-1], points[:-1, 1:], points[1:, :-1]
    normal = np.cross(p01 - p00, p10 - p00)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
    shade = 0.38 + 0.62 * np.clip(normal @ light, 0.0, 1.0)
    face_rgb = np.clip(rgb[:-1, :-1] * shade[..., None], 0.0, 1.0)
    fill = (face_rgb * 255.0).astype(np.uint8)

    width_px, height_px = size
    focal = _focal(size)
    screen_x = pitched[..., 0] / view_z * focal + width_px * 0.5
    screen_y = -pitched[..., 1] / view_z * focal + height_px * 0.5
    depth = (
        view_z[:-1, :-1] + view_z[:-1, 1:] + view_z[1:, :-1] + view_z[1:, 1:]
    ) * 0.25
    order = np.argsort(depth, axis=None)[::-1].astype(np.int32)
    pixels = _rasterize_mesh(
        np.ascontiguousarray(screen_x, dtype=np.float32),
        np.ascontiguousarray(screen_y, dtype=np.float32),
        np.ascontiguousarray(depth, dtype=np.float32),
        np.ascontiguousarray(fill),
        order,
        np.int32(width_px),
        np.int32(height_px),
    )
    return Image.fromarray(pixels, mode="RGB")


@njit(cache=True)
def _rasterize_mesh(screen_x, screen_y, depth, fill, order, width, height):
    """Z-buffered quads. Farther triangles are drawn first via `order`."""
    image = np.empty((height, width, 3), np.uint8)
    image[:] = 30
    zbuf = np.full((height, width), np.float32(1.0e9))
    quad_cols = screen_x.shape[1] - 1
    limit = width * height
    for n in range(order.shape[0]):
        index = order[n]
        row = index // quad_cols
        col = index - row * quad_cols
        x0 = screen_x[row, col]
        y0 = screen_y[row, col]
        x1 = screen_x[row, col + 1]
        y1 = screen_y[row, col + 1]
        x2 = screen_x[row + 1, col + 1]
        y2 = screen_y[row + 1, col + 1]
        x3 = screen_x[row + 1, col]
        y3 = screen_y[row + 1, col]
        min_x = x0
        max_x = x0
        min_y = y0
        max_y = y0
        for value in (x1, x2, x3):
            if value < min_x:
                min_x = value
            if value > max_x:
                max_x = value
        for value in (y1, y2, y3):
            if value < min_y:
                min_y = value
            if value > max_y:
                max_y = value
        if max_x < 0.0 or max_y < 0.0 or min_x >= width or min_y >= height:
            continue
        box_w = int(max_x) - int(min_x)
        box_h = int(max_y) - int(min_y)
        if box_w > 512 or box_h > 512 or box_w * box_h > limit:
            continue
        ix0 = int(min_x)
        if ix0 < 0:
            ix0 = 0
        iy0 = int(min_y)
        if iy0 < 0:
            iy0 = 0
        ix1 = int(max_x) + 1
        if ix1 > width:
            ix1 = width
        iy1 = int(max_y) + 1
        if iy1 > height:
            iy1 = height
        z = depth[row, col]
        red = fill[row, col, 0]
        green = fill[row, col, 1]
        blue = fill[row, col, 2]
        _fill_triangle(image, zbuf, x0, y0, x1, y1, x2, y2, z, red, green, blue, ix0, iy0, ix1, iy1)
        _fill_triangle(image, zbuf, x0, y0, x2, y2, x3, y3, z, red, green, blue, ix0, iy0, ix1, iy1)
    return image


@njit(cache=True)
def _fill_triangle(image, zbuf, x0, y0, x1, y1, x2, y2, z, red, green, blue, ix0, iy0, ix1, iy1):
    area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    if area > -1.0e-4 and area < 1.0e-4:
        return
    for py in range(iy0, iy1):
        sy = py + 0.5
        for px in range(ix0, ix1):
            sx = px + 0.5
            w0 = (x1 - sx) * (y2 - sy) - (y1 - sy) * (x2 - sx)
            w1 = (x2 - sx) * (y0 - sy) - (y2 - sy) * (x0 - sx)
            w2 = (x0 - sx) * (y1 - sy) - (y0 - sy) * (x1 - sx)
            if (w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0) or (w0 <= 0.0 and w1 <= 0.0 and w2 <= 0.0):
                if z < zbuf[py, px]:
                    zbuf[py, px] = z
                    image[py, px, 0] = red
                    image[py, px, 1] = green
                    image[py, px, 2] = blue


class MeshView(tk.Toplevel):
    """Orbit a heightfield textured with the source image."""

    def __init__(self, master, color_chw, height_chw):
        super().__init__(master)
        self.title("Heightmap 3D")
        self.configure(bg="#1e1e1e")
        self.yaw = 0.7
        self.pitch = 0.55
        self.distance = 2.5
        self.height_scale = tk.DoubleVar(value=0.35)
        self._color = color_chw
        self._height = height_chw
        self._photo = None
        self._drag_origin = None
        self._refine_job = None
        self._fine = True
        self._busy = False
        self._stale = False
        self._frames = queue.Queue()
        self._size = (900, 640)

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=12, pady=8)
        ttk.Label(bar, text="Height", width=10).pack(side="left")
        ttk.Scale(
            bar,
            from_=0.05,
            to=1.2,
            variable=self.height_scale,
            command=self._on_height,
        ).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Label(bar, text="Drag to orbit. Scroll to zoom.").pack(side="left")

        self.canvas = tk.Canvas(
            self,
            width=self._size[0],
            height=self._size[1],
            bg="#1e1e1e",
            highlightthickness=0,
        )
        self.canvas.pack(padx=12, pady=(0, 12))
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<MouseWheel>", self._zoom)
        self.after(1, lambda: self._request_draw(fine=True))
        self.after(16, self._poll_frames)

    def _on_height(self, _value):
        # The scale updates immediately. Rendering catches up on a background thread.
        self._redraw_then_refine()

    def set_maps(self, color_chw, height_chw):
        self._color = color_chw
        self._height = height_chw
        self._request_draw(fine=True)

    def _press(self, event):
        self._cancel_refine()
        self._drag_origin = (event.x, event.y, self.yaw, self.pitch)

    def _drag(self, event):
        if self._drag_origin is None:
            return
        x0, y0, yaw0, pitch0 = self._drag_origin
        self.yaw = yaw0 + (event.x - x0) * 0.01
        self.pitch = float(np.clip(pitch0 + (event.y - y0) * 0.01, -1.2, 1.2))
        self._request_draw(fine=False)

    def _release(self, _event):
        self._drag_origin = None
        self._request_draw(fine=True)

    def _zoom(self, event):
        self.distance *= 0.85 if event.delta > 0 else 1.18
        self.distance = float(np.clip(self.distance, 0.12, 8.0))
        self._redraw_then_refine()

    def _cancel_refine(self):
        if self._refine_job is not None:
            self.after_cancel(self._refine_job)
            self._refine_job = None

    def _redraw_then_refine(self):
        self._cancel_refine()
        self._request_draw(fine=False)
        self._refine_job = self.after(160, lambda: self._request_draw(fine=True))

    def _request_draw(self, fine):
        if fine:
            self._fine = True
        else:
            self._fine = False
        self._stale = True
        if not self._busy:
            self._start_render()

    def _start_render(self):
        self._busy = True
        self._stale = False
        fine = self._fine
        self._fine = False
        threading.Thread(
            target=self._render_worker,
            args=(
                self._color,
                self._height,
                self.yaw,
                self.pitch,
                self.distance,
                float(self.height_scale.get()),
                0 if fine else 180,
                self._size,
            ),
            daemon=True,
        ).start()

    def _render_worker(self, color, height, yaw, pitch, distance, height_scale, target_grid, size):
        image = None
        try:
            image = render_relief(color, height, yaw, pitch, distance, height_scale, target_grid, size)
        except Exception:
            image = None
        self._frames.put(image)

    def _poll_frames(self):
        if not self.winfo_exists():
            return
        try:
            while True:
                image = self._frames.get_nowait()
                self._busy = False
                if image is not None:
                    self._photo = ImageTk.PhotoImage(image)
                    self.canvas.delete("all")
                    self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
                if self._stale:
                    self._start_render()
        except queue.Empty:
            pass
        self.after(16, self._poll_frames)


class HeightmapApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DeepBump heightmap")
        self.minsize(520, 640)
        self._height = None
        self._color = None
        self._filtered = None
        self._filtered_amount = None
        self._view = None
        self._photo = None
        self._worker = None
        self._events = queue.Queue()

        self.input_path = tk.StringVar()
        self.overlap = tk.StringVar(value="LARGE")
        self.seamless = tk.BooleanVar(value=False)
        self.highpass = tk.DoubleVar(value=0.0)
        self.status = tk.StringVar(value="Choose an image.")

        self._build()
        self.after(100, self._poll_events)

    def _build(self):
        pad = {"padx": 12, "pady": 6}
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Image").grid(row=0, column=0, sticky="w", **pad)
        path_row = ttk.Frame(frm)
        path_row.grid(row=0, column=1, sticky="ew", **pad)
        path_row.columnconfigure(0, weight=1)
        ttk.Entry(path_row, textvariable=self.input_path).grid(row=0, column=0, sticky="ew")
        ttk.Button(path_row, text="Browse", command=self._browse).grid(row=0, column=1, padx=(8, 0))

        ttk.Label(frm, text="Tile overlap").grid(row=1, column=0, sticky="w", **pad)
        opts = ttk.Frame(frm)
        opts.grid(row=1, column=1, sticky="w", **pad)
        ttk.Combobox(
            opts,
            textvariable=self.overlap,
            values=("SMALL", "MEDIUM", "LARGE"),
            state="readonly",
            width=12,
        ).pack(side="left")
        ttk.Checkbutton(opts, text="Seamless", variable=self.seamless).pack(side="left", padx=(16, 0))

        actions = ttk.Frame(frm)
        actions.grid(row=2, column=0, columnspan=2, sticky="w", **pad)
        self.generate_btn = ttk.Button(actions, text="Generate heightmap", command=self._generate)
        self.generate_btn.pack(side="left")
        self.view_btn = ttk.Button(actions, text="3D view", command=self._open_view, state="disabled")
        self.view_btn.pack(side="left", padx=(8, 0))
        self.export_btn = ttk.Button(actions, text="Export PNG", command=self._export, state="disabled")
        self.export_btn.pack(side="left", padx=(8, 0))

        filt = ttk.Frame(frm)
        filt.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)
        ttk.Label(filt, text="High pass", width=12).pack(side="left")
        ttk.Scale(
            filt,
            from_=0,
            to=100,
            variable=self.highpass,
            command=self._on_highpass,
        ).pack(side="left", fill="x", expand=True, padx=8)
        self.highpass_readout = ttk.Label(filt, text="off", width=6)
        self.highpass_readout.pack(side="left")

        ttk.Label(frm, textvariable=self.status, wraplength=480).grid(
            row=4, column=0, columnspan=2, sticky="w", **pad
        )

        self.preview = tk.Canvas(frm, width=PREVIEW_MAX, height=PREVIEW_MAX, bg="#1e1e1e", highlightthickness=0)
        self.preview.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 12))
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(5, weight=1)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.input_path.set(path)

    def _generate(self):
        path = self.input_path.get().strip()
        if not path:
            messagebox.showwarning("DeepBump", "Choose an image first.")
            return
        if self._worker and self._worker.is_alive():
            return
        self.generate_btn.state(["disabled"])
        self.view_btn.state(["disabled"])
        self.export_btn.state(["disabled"])
        self.status.set("Generating normals…")
        overlap = self.overlap.get()
        seamless = self.seamless.get()
        self._worker = threading.Thread(
            target=self._run_pipeline,
            args=(path, overlap, seamless),
            daemon=True,
        )
        self._worker.start()

    def _run_pipeline(self, path, overlap, seamless):
        try:
            color = load_color_image(path)
            normals = module_color_to_normals.apply(color, overlap, None)
            self._events.put(("status", "Generating height…"))
            height = module_normals_to_height.apply(normals, seamless, None)
            self._events.put(("done", (height, color)))
        except Exception as exc:
            self._events.put(("error", str(exc)))

    def _poll_events(self):
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "status":
                    self.status.set(payload)
                elif kind == "done":
                    height, color = payload
                    self._height = height
                    self._color = color
                    self._filtered = None
                    self._filtered_amount = None
                    shown = self._output_height()
                    self._show_preview(shown)
                    h, w = height.shape[1], height.shape[2]
                    self.status.set(f"Heightmap ready ({w}×{h}). Export writes an 8-bit PNG.")
                    self.generate_btn.state(["!disabled"])
                    self.view_btn.state(["!disabled"])
                    self.export_btn.state(["!disabled"])
                    if self._view is not None and self._view.winfo_exists():
                        self._view.set_maps(color, shown)
                elif kind == "error":
                    self.status.set("Failed.")
                    self.generate_btn.state(["!disabled"])
                    if self._height is not None:
                        self.view_btn.state(["!disabled"])
                        self.export_btn.state(["!disabled"])
                    messagebox.showerror("DeepBump", payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _show_preview(self, height_chw):
        gray = height_to_u8(height_chw)
        h, w = gray.shape
        step = max(1, int(np.ceil(max(h, w) / PREVIEW_MAX)))
        small = np.ascontiguousarray(gray[::step, ::step])
        rgb = np.stack([small, small, small], axis=-1)
        with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as handle:
            handle.write(f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode("ascii"))
            handle.write(rgb.tobytes())
            preview_path = handle.name
        self._photo = tk.PhotoImage(file=preview_path)
        os.remove(preview_path)
        self.preview.delete("all")
        self.preview.create_image(PREVIEW_MAX // 2, PREVIEW_MAX // 2, image=self._photo)

    def _on_highpass(self, _value):
        amount = float(self.highpass.get())
        self.highpass_readout.configure(text="off" if amount < 0.5 else str(int(round(amount))))
        if self._height is None:
            return
        shown = self._output_height()
        self._show_preview(shown)
        if self._view is not None and self._view.winfo_exists():
            self._view.set_maps(self._color, shown)

    def _output_height(self):
        amount = int(round(float(self.highpass.get())))
        if amount <= 0 or self._height is None:
            return self._height
        if self._filtered is None or self._filtered_amount != amount:
            self._filtered = highpass_height(self._height, amount)
            self._filtered_amount = amount
        return self._filtered

    def _open_view(self):
        if self._height is None or self._color is None:
            return
        shown = self._output_height()
        if self._view is None or not self._view.winfo_exists():
            self._view = MeshView(self, self._color, shown)
        else:
            self._view.set_maps(self._color, shown)
            self._view.lift()

    def _export(self):
        if self._height is None:
            return
        path = filedialog.asksaveasfilename(
            title="Export heightmap",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        gray = height_to_u8(self._output_height())
        rgb = np.stack([gray, gray, gray], axis=-1)
        try:
            iio.imwrite(path, rgb)
        except Exception as exc:
            messagebox.showerror("DeepBump", str(exc))
            return
        self.status.set(f"Saved {path}")


if __name__ == "__main__":
    HeightmapApp().mainloop()
