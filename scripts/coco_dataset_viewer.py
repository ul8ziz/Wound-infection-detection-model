#!/usr/bin/env python3
"""COCO Dataset Viewer for Wound Detection Project.

View images with bboxes and masks from COCO-format annotations.
Supports --check mode to validate dataset and report issues.

Usage:
    cd scripts
    python coco_dataset_viewer.py
    python coco_dataset_viewer.py -i ../data/wound_focus_clean -a ../data/wound_focus_clean/val_wound_only.json
    python coco_dataset_viewer.py --check

Keyboard: Left/Right or j/k to navigate, Home/End for first/last image, b/l/m to toggle bboxes/labels/masks, space to toggle all.
File > Load annotations... (Ctrl+O) to switch to a different JSON file.
Click an image in the left panel to jump to it.
"""
import argparse
import colorsys
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

# Tkinter imports (optional for --check mode)
try:
    import tkinter as tk
    import tkinter.ttk as ttk
    from tkinter import filedialog, messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
# Default dataset: standardized wound-only splits (images under <root>/images/...)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "wound_focus_clean"
DEFAULT_ANNOTATIONS = DEFAULT_DATA_ROOT / "val_wound_only.json"

parser = argparse.ArgumentParser(
    description="View images with bboxes/masks from COCO dataset, or validate with --check"
)
parser.add_argument(
    "-i", "--images",
    default="",
    type=str,
    metavar="PATH",
    help="Path to images root (e.g. ../data/wound_focus_clean; file_name is relative to this)",
)
parser.add_argument(
    "-a", "--annotations",
    default="",
    type=str,
    metavar="PATH",
    help="Path to annotations JSON (e.g. ../data/wound_focus_clean/val_wound_only.json)",
)
parser.add_argument(
    "--check",
    action="store_true",
    help="Validate dataset only (no GUI); report issues",
)


class Data:
    """Handles data loading and preparation."""

    def __init__(self, image_dir: str, annotations_file: str):
        self.image_dir = Path(image_dir) if image_dir else None
        instances, images, categories = parse_coco(annotations_file)
        self.instances = instances
        self.images = ImageList(images)
        self.categories = categories
        self.current_image = self.images.next()

    def prepare_image(self, object_based_coloring: bool = False):
        """Prepares image path, objects, and colors for current image."""
        img_id, img_name = self.current_image
        full_path = resolve_coco_image_path(self.image_dir, img_name) if self.image_dir else None

        objects = [obj for obj in self.instances["annotations"] if obj["image_id"] == img_id]
        obj_categories_ids = [obj["category_id"] for obj in objects]
        img_obj_categories = [obj["category_id"] for obj in objects]
        img_categories = sorted(list(set(img_obj_categories)))
        names_colors = [self.categories.get(i, ["?", (128, 128, 128)]) for i in obj_categories_ids]

        json_w, json_h = None, None
        for img in self.instances.get("images", []):
            if img.get("id") == img_id:
                json_w = img.get("width")
                json_h = img.get("height")
                break

        if object_based_coloring:
            obj_colors = prepare_colors(len(objects))
            names_colors = [[names_colors[i][0], obj_colors[i]] for i in range(len(objects))]

        return full_path, objects, names_colors, img_obj_categories, img_categories, json_w, json_h

    def next_image(self):
        self.current_image = self.images.next()

    def previous_image(self):
        self.current_image = self.images.prev()

    def go_to_image(self, idx: int) -> bool:
        """Jump to image at index. Returns True if successful."""
        result = self.images.go_to_index(idx)
        if result is not None:
            self.current_image = result
            return True
        return False


def parse_coco(annotations_file: str) -> tuple:
    """Parses COCO JSON annotation file."""
    instances = load_annotations(annotations_file)
    images = get_images(instances)
    categories = get_categories(instances)
    return instances, images, categories


def load_annotations(fname: str) -> dict:
    """Loads annotations file."""
    path = Path(fname)
    if not path.is_absolute():
        path = (SCRIPT_DIR / fname).resolve()
    logging.info(f"Parsing {path}...")
    with open(path, encoding="utf-8") as f:
        instances = json.load(f)
    return instances


def resolve_coco_image_path(image_dir: Path, file_name: str) -> str:
    """Join COCO ``file_name`` to ``image_dir``.

    If the JSON stores paths like ``original_data/task_0/data/x.jpg`` but the user
    selects ``.../original_data`` as the images root, the naive join would look for
    ``.../original_data/original_data/...``. Strip leading segments while they repeat
    ``image_dir.name`` so both layouts work.
    """
    if not image_dir or not file_name:
        return ""
    fn = file_name.replace("\\", "/").strip()
    root_name = image_dir.name
    rel = fn
    while True:
        candidate = image_dir.joinpath(*[p for p in rel.split("/") if p])
        if candidate.exists():
            return str(candidate.resolve())
        parts = rel.split("/", 1)
        if len(parts) < 2 or parts[0] != root_name:
            return str(image_dir.joinpath(*[p for p in fn.split("/") if p]).resolve())
        rel = parts[1]


def resolve_annotation_path(ann: str) -> Path:
    """Resolve annotations path the same way as load_annotations (for existence checks)."""
    path = Path(ann)
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path


def get_images(instances: dict) -> list:
    """Extracts image ids and file names from annotations."""
    return [(img["id"], img["file_name"]) for img in instances["images"]]


def open_image(full_img_path: str):
    """Opens image and creates draw context."""
    img_open = Image.open(full_img_path).convert("RGBA")
    draw_layer = Image.new("RGBA", img_open.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(draw_layer)
    return img_open, draw_layer, draw


def prepare_colors(n_objects: int, shuffle: bool = True) -> list:
    """Generate distinct colors for objects."""
    if n_objects <= 0:
        return []
    hsv_tuples = [(x / max(1, n_objects), 1.0, 1.0) for x in range(n_objects)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colors]
    if shuffle:
        random.seed(42)
        random.shuffle(colors)
        random.seed(None)
    return colors


def get_categories(instances: dict) -> dict:
    """Extracts categories and assigns color to each."""
    cats = instances.get("categories", [])
    n = max(len(cats), 1)
    colors = prepare_colors(n, shuffle=True)
    categories = {}
    for i, cat in enumerate(cats):
        cid = cat["id"]
        name = cat.get("name", "?")
        color = colors[i % len(colors)]
        categories[cid] = [name, color]
    return categories


def _scale_objects(objects, scale_x: float, scale_y: float):
    """Scale annotation coordinates to match actual image dimensions. Returns new objects (no mutation)."""
    if scale_x == 1.0 and scale_y == 1.0:
        return objects
    scaled = []
    for obj in objects:
        o = dict(obj)
        b = o.get("bbox")
        if b:
            o["bbox"] = [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
        seg = o.get("segmentation")
        if isinstance(seg, list):
            o["segmentation"] = []
            for poly in seg:
                if poly and len(poly) >= 6:
                    scaled_poly = []
                    for j in range(0, len(poly), 2):
                        scaled_poly.append(poly[j] * scale_x)
                        scaled_poly.append(poly[j + 1] * scale_y)
                    o["segmentation"].append(scaled_poly)
        scaled.append(o)
    return scaled


def draw_bboxes(draw, objects, labels, obj_categories, ignore, width, label_size):
    """Draws bounding boxes on the image."""
    bboxes = [
        [obj["bbox"][0], obj["bbox"][1], obj["bbox"][0] + obj["bbox"][2], obj["bbox"][1] + obj["bbox"][3]]
        for obj in objects
    ]
    for i, (c, b) in enumerate(zip(obj_categories, bboxes)):
        if i not in ignore:
            draw.rectangle(b, outline=c[-1], width=width)
            if labels and c[0]:
                try:
                    font = ImageFont.truetype("arial.ttf", size=label_size)
                except OSError:
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=label_size)
                    except OSError:
                        font = ImageFont.load_default()
                text = c[0]
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except AttributeError:
                    tw, th = draw.textsize(text, font=font)
                tx0, ty0 = b[0], max(0, b[1] - th)
                tx1, ty1 = tx0 + tw, ty0 + th
                draw.rectangle((tx0, ty0, tx1, ty1), fill=c[-1])
                draw.text((tx0, ty0), text, (255, 255, 255), font=font)


def draw_masks(draw, objects, obj_categories, ignore, alpha):
    """Draws segmentation masks over image."""
    for i, obj in enumerate(objects):
        if i in ignore:
            continue
        m = obj.get("segmentation")
        if not m:
            continue
        c = obj_categories[i] if i < len(obj_categories) else ["?", (128, 128, 128)]
        fill = tuple(list(c[-1]) + [alpha])
        if isinstance(m, list):
            for poly in m:
                if poly and len(poly) >= 6:
                    flat = [(poly[j], poly[j + 1]) for j in range(0, len(poly), 2)]
                    draw.polygon(flat, outline=fill, fill=fill)
        elif isinstance(m, dict) and obj.get("iscrowd"):
            try:
                mask = rle_to_mask(m.get("counts"), m.get("size", [0, 0])[0], m.get("size", [0, 0])[1])
                if mask is not None:
                    mask_img = Image.fromarray(mask)
                    draw.bitmap((0, 0), mask_img, fill=fill)
            except Exception:
                pass


def rle_to_mask(rle, height, width):
    """Converts RLE to binary mask (simplified; pycocotools format may differ)."""
    if rle is None or height <= 0 or width <= 0:
        return None
    try:
        if isinstance(rle, dict):
            from pycocotools import mask as mask_util
            return mask_util.decode(rle)
        if isinstance(rle, (list, np.ndarray)):
            rle = np.array(rle)
            if rle.size % 2 != 0:
                return None
            pairs = rle.reshape(-1, 2)
            img = np.zeros(height * width, dtype=np.uint8)
            idx = 0
            for start, length in pairs:
                idx += int(start)
                end = min(idx + int(length), len(img))
                img[idx:end] = 255
                idx += int(length)
            return img.reshape(height, width).T
    except Exception:
        pass
    return None


class ImageList:
    """Handles image navigation."""

    def __init__(self, images: list):
        self.image_list = images or []
        self.n = -1
        self.max = len(self.image_list)

    def next(self):
        self.n += 1
        if self.n >= self.max:
            self.n = 0
        return self.image_list[self.n]

    def prev(self):
        if self.n <= 0:
            self.n = self.max - 1
        else:
            self.n -= 1
        return self.image_list[self.n]

    def go_to_index(self, idx: int):
        """Jump to image at index (0-based)."""
        if 0 <= idx < self.max:
            self.n = idx
            return self.image_list[self.n]
        return None

    def get_current_index(self) -> int:
        """Returns current image index (0-based)."""
        return self.n


# ---------------------------------------------------------------------------
# Dataset validation (--check mode)
# ---------------------------------------------------------------------------

def check_dataset(annotations_path: str, images_root: str) -> int:
    """Validates COCO dataset and reports issues. Returns number of issues."""
    path = Path(annotations_path)
    if not path.is_absolute():
        path = (SCRIPT_DIR / annotations_path).resolve()
    if not path.exists():
        logging.error(f"Annotations not found: {path}")
        return 1

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    categories = {c["id"]: c["name"] for c in data["categories"]}
    anns = data["annotations"]
    img_root = Path(images_root) if images_root else (SCRIPT_DIR.parent / "data")
    if not img_root.is_absolute():
        img_root = (SCRIPT_DIR / img_root).resolve()

    issues = 0

    # Missing images
    missing = []
    for img_id, img in images.items():
        fn = img.get("file_name", "")
        resolved = resolve_coco_image_path(img_root, fn)
        if not resolved or not Path(resolved).exists():
            missing.append(fn)
    if missing:
        logging.warning(f"Missing images ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        issues += len(missing)

    # Empty segmentations
    empty_seg = [a for a in anns if not a.get("segmentation") or len(a["segmentation"]) == 0]
    if empty_seg:
        logging.warning(f"Empty segmentations: {len(empty_seg)} annotations")
        issues += len(empty_seg)

    # Invalid bboxes
    bad_bbox = []
    for a in anns:
        bbox = a.get("bbox", [])
        if len(bbox) != 4:
            bad_bbox.append(a.get("id"))
            continue
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            bad_bbox.append(a.get("id"))
    if bad_bbox:
        logging.warning(f"Invalid bboxes (w/h<=0): {len(bad_bbox)} annotations")
        issues += len(bad_bbox)

    # Category not in categories
    cat_ids = set(categories.keys())
    unknown = [a for a in anns if a.get("category_id") not in cat_ids]
    if unknown:
        logging.warning(f"Unknown category_id: {len(unknown)} annotations")
        issues += len(unknown)

    # Summary
    logging.info(f"Dataset: {len(images)} images, {len(anns)} annotations, {len(categories)} categories")
    if issues == 0:
        logging.info("No issues found.")
    else:
        logging.warning(f"Total issues: {issues}")

    return issues


# ---------------------------------------------------------------------------
# GUI components (only if tkinter available)
# ---------------------------------------------------------------------------

if HAS_TK:

    class ImagePanel(ttk.Frame):
        """Canvas panel for displaying images with scrollbars."""

        def __init__(self, parent, width=768, height=480, canvwidth=600, canvheight=500):
            super().__init__(parent, width=width, height=height)
            self._rootwindow = self.winfo_toplevel()
            self.width, self.height = width, height
            self.canvwidth, self.canvheight = canvwidth, canvheight
            self.bg = "gray15"
            self.pack(fill=tk.BOTH, expand=True)
            self.image = None

            self._canvas = tk.Canvas(self, width=width, height=height, bg=self.bg, relief="sunken", borderwidth=2)
            self.hscroll = ttk.Scrollbar(self, command=self._canvas.xview, orient=tk.HORIZONTAL)
            self.vscroll = ttk.Scrollbar(self, command=self._canvas.yview)
            self._canvas.configure(xscrollcommand=self.hscroll.set, yscrollcommand=self.vscroll.set)

            self.rowconfigure(0, weight=1, minsize=0)
            self.columnconfigure(0, weight=1, minsize=0)
            self._canvas.grid(row=0, column=0, rowspan=1, columnspan=1, sticky=tk.NSEW, padx=1, pady=1)
            self.vscroll.grid(row=0, column=1, rowspan=1, columnspan=1, sticky=tk.NSEW, padx=1, pady=1)
            self.hscroll.grid(row=1, column=0, rowspan=1, columnspan=1, sticky=tk.NSEW, padx=1, pady=1)

            self.reset()
            self._rootwindow.bind("<Configure>", self.on_resize)

        def create_image(self, *args, **kwargs):
            return self._canvas.create_image(*args, **kwargs)

        def delete(self, *args):
            return self._canvas.delete(*args)

        def bind(self, *args, **kwargs):
            return self._canvas.bind(*args, **kwargs)

        def focus_force(self):
            return self._canvas.focus_force()

        def reset(self, canvwidth=None, canvheight=None, bg=None):
            if canvwidth:
                self.canvwidth = canvwidth
            if canvheight:
                self.canvheight = canvheight
            if bg:
                self.bg = bg
            self._canvas.config(
                bg=self.bg,
                scrollregion=(0, 0, self.canvwidth, self.canvheight),
            )
            self._canvas.xview_moveto(0)
            self._canvas.yview_moveto(0)
            self.adjust_scrolls()

        def adjust_scrolls(self):
            cwidth = self._canvas.winfo_width()
            cheight = self._canvas.winfo_height()
            if cwidth < self.canvwidth:
                self._canvas.xview_moveto(0)
            if cheight < self.canvheight:
                self._canvas.yview_moveto(0)
            if cwidth < self.canvwidth:
                self.hscroll.grid(row=1, column=0, sticky=tk.NSEW, padx=1, pady=1)
            else:
                self.hscroll.grid_forget()
            if cheight < self.canvheight:
                self.vscroll.grid(row=0, column=1, sticky=tk.NSEW, padx=1, pady=1)
            else:
                self.vscroll.grid_forget()

        def on_resize(self, event):
            self.adjust_scrolls()

    class StatusBar(ttk.Frame):
        def __init__(self, parent):
            super().__init__(parent)
            self.pack(side=tk.BOTTOM, fill=tk.X)
            self.file_count = ttk.Label(self, borderwidth=5, background="gray75")
            self.file_count.pack(side=tk.RIGHT)
            self.description = ttk.Label(self, borderwidth=5, background="gray75")
            self.description.pack(side=tk.RIGHT)
            self.file_name = ttk.Label(self, borderwidth=5, background="gray75")
            self.file_name.pack(side=tk.LEFT)
            self.nobjects = ttk.Label(self, borderwidth=5, background="gray75")
            self.nobjects.pack(side=tk.LEFT)
            self.ncategories = ttk.Label(self, borderwidth=5, background="gray75")
            self.ncategories.pack(side=tk.LEFT)

    class Menu(tk.Menu):
        def __init__(self, parent):
            super().__init__(parent)
            self.file = tk.Menu(self, tearoff=False)
            self.file.add_command(label="Load annotations...", accelerator="Ctrl+O")
            self.file.add_command(label="Change images folder...")
            self.file.add_command(label="Save", accelerator="Ctrl+S")
            self.file.add_separator()
            self.file.add_command(label="Exit", accelerator="Ctrl+Q")
            self.add_cascade(label="File", menu=self.file)
            self.view = tk.Menu(self, tearoff=False)
            self.view.add_checkbutton(label="BBoxes", onvalue=True, offvalue=False)
            self.view.add_checkbutton(label="Labels", onvalue=True, offvalue=False)
            self.view.add_checkbutton(label="Masks", onvalue=True, offvalue=False)
            self.view.add_checkbutton(label="Fit to window (full image)", onvalue=True, offvalue=False)
            self.add_cascade(label="View", menu=self.view)
            self.colormenu = tk.Menu(self.view, tearoff=0)
            self.colormenu.add_radiobutton(label="Categories", value=False)
            self.colormenu.add_radiobutton(label="Objects", value=True)
            self.view.add_cascade(label="Coloring", menu=self.colormenu)

    class ImagesPanel(ttk.Frame):
        """Panel with scrollable list of all images for selection and navigation."""

        def __init__(self, parent):
            super().__init__(parent)
            self.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=2)
            ttk.Label(self, text="Images (click to go)", borderwidth=2, background="gray50").pack(side=tk.TOP, fill=tk.X)
            scroll = ttk.Scrollbar(self)
            self.image_listbox = tk.Listbox(self, selectmode=tk.SINGLE, exportselection=0, height=15, yscrollcommand=scroll.set)
            scroll.config(command=self.image_listbox.yview)
            self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scroll.pack(side=tk.RIGHT, fill=tk.Y)
            nav_frame = ttk.Frame(self)
            self.btn_prev = ttk.Button(nav_frame, text="< Prev")
            self.btn_prev.pack(side=tk.LEFT, padx=2)
            self.btn_next = ttk.Button(nav_frame, text="Next >")
            self.btn_next.pack(side=tk.LEFT, padx=2)
            nav_frame.pack(side=tk.BOTTOM, pady=2)

    class ObjectsPanel(ttk.PanedWindow):
        def __init__(self, parent):
            super().__init__(parent)
            self.pack(side=tk.RIGHT, fill=tk.Y)
            self.category_subpanel = ttk.Frame()
            ttk.Label(self.category_subpanel, text="categories", borderwidth=2, background="gray50").pack(side=tk.TOP, fill=tk.X)
            self.category_box = tk.Listbox(self.category_subpanel, selectmode=tk.EXTENDED, exportselection=0)
            self.category_box.pack(side=tk.TOP, fill=tk.Y, expand=True)
            self.add(self.category_subpanel)
            self.object_subpanel = ttk.Frame()
            ttk.Label(self.object_subpanel, text="objects", borderwidth=2, background="gray50").pack(side=tk.TOP, fill=tk.X)
            self.object_box = tk.Listbox(self.object_subpanel, selectmode=tk.EXTENDED, exportselection=0)
            self.object_box.pack(side=tk.TOP, fill=tk.Y, expand=True)
            self.add(self.object_subpanel)

    class SlidersBar(ttk.Frame):
        def __init__(self, parent):
            super().__init__(parent)
            self.pack(side=tk.BOTTOM, fill=tk.X)
            self.bbox_slider = tk.Scale(self, label="bbox", from_=0, to=25, tickinterval=5, orient=tk.HORIZONTAL)
            self.bbox_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.label_slider = tk.Scale(self, label="label", from_=10, to=100, tickinterval=25, orient=tk.HORIZONTAL)
            self.label_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.mask_slider = tk.Scale(self, label="mask", from_=0, to=255, tickinterval=50, orient=tk.HORIZONTAL)
            self.mask_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

    class Controller:
        def __init__(self, data, root, image_panel, statusbar, menu, objects_panel, images_panel, sliders, annotations_path=""):
            self.data = data
            self._annotations_path = annotations_path
            self.root = root
            self.image_panel = image_panel
            self.statusbar = statusbar
            self.menu = menu
            self.objects_panel = objects_panel
            self.images_panel = images_panel
            self.sliders = sliders

            self.file_count_status = tk.StringVar()
            self.file_name_status = tk.StringVar()
            self.description_status = tk.StringVar()
            self.nobjects_status = tk.StringVar()
            self.ncategories_status = tk.StringVar()
            self.statusbar.file_count.configure(textvariable=self.file_count_status)
            self.statusbar.file_name.configure(textvariable=self.file_name_status)
            self.statusbar.description.configure(textvariable=self.description_status)
            self.statusbar.nobjects.configure(textvariable=self.nobjects_status)
            self.statusbar.ncategories.configure(textvariable=self.ncategories_status)

            self.bboxes_on_global = tk.BooleanVar(value=True)
            self.labels_on_global = tk.BooleanVar(value=True)
            self.masks_on_global = tk.BooleanVar(value=True)
            self.coloring_on_global = tk.BooleanVar(value=False)
            self.fit_to_window = tk.BooleanVar(value=True)

            self.menu.file.entryconfigure("Load annotations...", command=self.load_annotations)
            self.menu.file.entryconfigure("Change images folder...", command=self.change_images_folder)
            self.menu.file.entryconfigure("Save", command=self.save_image)
            self.menu.file.entryconfigure("Exit", command=self.exit)
            self.menu.view.entryconfigure("BBoxes", variable=self.bboxes_on_global, command=self.menu_view_bboxes)
            self.menu.view.entryconfigure("Labels", variable=self.labels_on_global, command=self.menu_view_labels)
            self.menu.view.entryconfigure("Masks", variable=self.masks_on_global, command=self.menu_view_masks)
            self.menu.view.entryconfigure("Fit to window (full image)", variable=self.fit_to_window, command=self.menu_view_fit)
            self.menu.colormenu.entryconfigure("Categories", variable=self.coloring_on_global, command=self.menu_view_coloring)
            self.menu.colormenu.entryconfigure("Objects", variable=self.coloring_on_global, command=self.menu_view_coloring)
            self.root.configure(menu=self.menu)

            self.bboxes_on_local = True
            self.labels_on_local = True
            self.masks_on_local = True
            self.coloring_on_local = False
            self.selected_cats = None
            self.selected_objs = None

            self.bbox_thickness = tk.IntVar(value=3)
            self.label_size = tk.IntVar(value=15)
            self.mask_alpha = tk.IntVar(value=128)
            self.sliders.bbox_slider.configure(variable=self.bbox_thickness, command=lambda e: self.update_img())
            self.sliders.label_slider.configure(variable=self.label_size, command=lambda e: self.update_img())
            self.sliders.mask_slider.configure(variable=self.mask_alpha, command=lambda e: self.update_img())

            self.images_panel.btn_prev.configure(command=self.prev_img)
            self.images_panel.btn_next.configure(command=self.next_img)
            self.bind_events()
            self.current_composed_image = None
            self.current_img_obj_categories = None
            self.current_img_categories = None
            self.update_img()

        def compose_image(self, full_path, objects, names_colors, json_w=None, json_h=None, bboxes_on=True, labels_on=True, masks_on=True, ignore=None, width=1, alpha=128, label_size=15):
            ignore = ignore or []
            if not full_path or not os.path.exists(full_path):
                return
            img_open, draw_layer, draw = open_image(full_path)
            actual_w, actual_h = img_open.size
            if json_w and json_h and json_w > 0 and json_h > 0 and (json_w != actual_w or json_h != actual_h):
                scale_x = actual_w / json_w
                scale_y = actual_h / json_h
                objects = _scale_objects(objects, scale_x, scale_y)
            if masks_on:
                draw_masks(draw, objects, names_colors, ignore, alpha)
            if bboxes_on:
                draw_bboxes(draw, objects, labels_on, names_colors, ignore, width, label_size)
            del draw
            self.current_composed_image = Image.alpha_composite(img_open, draw_layer)

        def update_img(self, local=True, width=None, alpha=None, label_size=None):
            full_path, objects, names_colors, img_obj_categories, img_categories, json_w, json_h = self.data.prepare_image(self.coloring_on_local)
            self.current_img_obj_categories = img_obj_categories
            self.current_img_categories = img_categories

            ignore = [] if self.selected_objs is None else [i for i in range(len(img_obj_categories)) if i not in self.selected_objs]
            width = self.bbox_thickness.get() if width is None else width
            alpha = self.mask_alpha.get() if alpha is None else alpha
            label_size = self.label_size.get() if label_size is None else label_size

            self.compose_image(full_path, objects, names_colors, json_w, json_h, self.bboxes_on_local, self.labels_on_local, self.masks_on_local, ignore, width, alpha, label_size)

            if self.current_composed_image is None:
                return
            img = self.current_composed_image
            w, h = img.size
            if self.fit_to_window.get():
                cw = max(100, self.image_panel._canvas.winfo_width() or 800)
                ch = max(100, self.image_panel._canvas.winfo_height() or 600)
                scale = min(cw / w, ch / h, 1.0)
                if scale < 1.0:
                    nw, nh = int(w * scale), int(h * scale)
                    try:
                        resample = Image.Resampling.LANCZOS
                    except AttributeError:
                        resample = Image.LANCZOS
                    img = img.resize((nw, nh), resample)
                w, h = img.size
            img_tk = ImageTk.PhotoImage(img)
            self.image_panel.delete("all")
            self.image_panel.create_image(0, 0, image=img_tk, anchor=tk.NW)
            self.image_panel.image = img_tk
            self.image_panel.reset(canvwidth=w, canvheight=h)

            self.file_count_status.set(f"{self.data.images.n + 1}/{self.data.images.max}")
            self.file_name_status.set(str(self.data.current_image[-1]))
            self.description_status.set(self.data.instances.get("info", {}).get("description", ""))
            self.nobjects_status.set(f"objects: {len(img_obj_categories)}")
            self.ncategories_status.set(f"categories: {len(img_categories)}")

            self.update_category_box()
            self.update_object_box()
            self.update_images_box()

        def update_images_box(self):
            """Populate and sync the images list; highlight current image."""
            self.images_panel.image_listbox.unbind("<<ListboxSelect>>")
            self.images_panel.image_listbox.delete(0, tk.END)
            for i, (img_id, fn) in enumerate(self.data.images.image_list):
                short = Path(fn).name if len(fn) > 40 else fn
                self.images_panel.image_listbox.insert(tk.END, f"{i+1}. {short}")
            idx = self.data.images.get_current_index()
            self.images_panel.image_listbox.selection_clear(0, tk.END)
            self.images_panel.image_listbox.selection_set(idx)
            self.images_panel.image_listbox.see(idx)
            self.images_panel.image_listbox.bind("<<ListboxSelect>>", self.select_image)

        def select_image(self, event=None):
            """Jump to selected image when user clicks in images list."""
            sel = self.images_panel.image_listbox.curselection()
            if sel:
                idx = int(sel[0])
                if self.data.go_to_image(idx):
                    self.set_locals()
                    self.selected_cats = None
                    self.selected_objs = None
                    self.update_img(local=False)
                    self.images_panel.image_listbox.selection_clear(0, tk.END)
                    self.images_panel.image_listbox.selection_set(idx)
                    self.images_panel.image_listbox.see(idx)

        def update_category_box(self):
            ids = self.current_img_categories
            names = [self.data.categories.get(i, ["?"])[0] for i in ids]
            items = [f"{i} {n}" for i, n in zip(ids, names)]
            self.objects_panel.category_box.delete(0, tk.END)
            for item in items:
                self.objects_panel.category_box.insert(tk.END, item)
            self.objects_panel.category_box.selection_clear(0, tk.END)
            if self.selected_cats is not None:
                for i in self.selected_cats:
                    self.objects_panel.category_box.select_set(i)
            else:
                self.objects_panel.category_box.select_set(0, tk.END)

        def update_object_box(self):
            ids = self.current_img_obj_categories
            names = [self.data.categories.get(i, ["?"])[0] for i in ids]
            items = [f"{i} {n}" for i, n in enumerate(names)]
            self.objects_panel.object_box.delete(0, tk.END)
            for item in items:
                self.objects_panel.object_box.insert(tk.END, item)
            self.objects_panel.object_box.selection_clear(0, tk.END)
            if self.selected_objs is not None:
                for i in self.selected_objs:
                    self.objects_panel.object_box.select_set(i)
            else:
                self.objects_panel.object_box.select_set(0, tk.END)

        def select_category(self, event):
            sel = self.objects_panel.category_box.curselection()
            self.selected_cats = list(sel)
            selected_objs = []
            for ci in self.selected_cats:
                for i, o in enumerate(self.current_img_obj_categories):
                    if self.current_img_categories[ci] == o:
                        selected_objs.append(i)
            self.selected_objs = selected_objs
            self.update_img()

        def select_object(self, event):
            sel = self.objects_panel.object_box.curselection()
            self.selected_objs = list(sel)
            selected_cats = []
            for oi in self.selected_objs:
                for i, c in enumerate(self.current_img_categories):
                    if self.current_img_obj_categories[oi] == c:
                        selected_cats.append(i)
            self.selected_cats = selected_cats
            self.update_img()

        def set_locals(self):
            self.bboxes_on_local = self.bboxes_on_global.get()
            self.labels_on_local = self.labels_on_global.get()
            self.masks_on_local = self.masks_on_global.get()
            self.coloring_on_local = self.coloring_on_global.get()

        def next_img(self, event=None):
            self.data.next_image()
            self.set_locals()
            self.selected_cats = None
            self.selected_objs = None
            self.update_img(local=False)

        def prev_img(self, event=None):
            self.data.previous_image()
            self.set_locals()
            self.selected_cats = None
            self.selected_objs = None
            self.update_img(local=False)

        def load_annotations(self, event=None):
            """Open a new annotations JSON file."""
            path = filedialog.askopenfilename(
                title="Select annotations JSON",
                filetypes=(("JSON files", "*.json"), ("all files", "*.*")),
                defaultextension=".json",
            )
            if not path:
                return
            try:
                img_dir = str(self.data.image_dir) if self.data.image_dir is not None else ""
                self._annotations_path = path
                self.data = Data(img_dir, path)
                self.set_locals()
                self.selected_cats = None
                self.selected_objs = None
                self.update_img()
                self.root.title(f"COCO Dataset Viewer - {Path(path).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load annotations:\n{e}")

        def change_images_folder(self, event=None):
            """Change the root folder for images."""
            folder = filedialog.askdirectory(title="Select images root folder")
            if not folder:
                return
            if not self._annotations_path:
                messagebox.showwarning("Warning", "No annotations loaded. Use Load annotations... first.")
                return
            try:
                self.data = Data(folder, self._annotations_path)
                self.set_locals()
                self.selected_cats = None
                self.selected_objs = None
                self.update_img()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to change folder:\n{e}")

        def save_image(self, event=None):
            initialfile = self.data.current_image[-1].split(".")[0]
            file = filedialog.asksaveasfilename(initialfile=initialfile, filetypes=(("png files", "*.png"), ("all files", "*.*")), defaultextension=".png")
            if file and self.current_composed_image:
                self.current_composed_image.save(file)

        def exit(self, event=None):
            self.root.quit()

        def menu_view_bboxes(self):
            self.bboxes_on_local = self.bboxes_on_global.get()
            self.update_img()

        def menu_view_labels(self):
            self.labels_on_local = self.labels_on_global.get()
            self.update_img()

        def menu_view_masks(self):
            self.masks_on_local = self.masks_on_global.get()
            self.update_img()

        def menu_view_coloring(self):
            self.coloring_on_local = self.coloring_on_global.get()
            self.update_img()

        def menu_view_fit(self):
            self.update_img()

        def toggle_bboxes(self, event=None):
            self.bboxes_on_local = not self.bboxes_on_local
            self.update_img()

        def toggle_labels(self, event=None):
            self.labels_on_local = not self.labels_on_local
            self.update_img()

        def toggle_masks(self, event=None):
            self.masks_on_local = not self.masks_on_local
            self.update_img()

        def toggle_all(self, event=None):
            if event and event.widget in (self.objects_panel.category_box, self.objects_panel.object_box, self.images_panel.image_listbox):
                return
            if any([self.bboxes_on_local, self.labels_on_local, self.masks_on_local]):
                self.bboxes_on_local = self.labels_on_local = self.masks_on_local = False
            else:
                self.bboxes_on_local = self.labels_on_local = self.masks_on_local = True
            self.update_img()

        def first_img(self, event=None):
            self.data.go_to_image(0)
            self.set_locals()
            self.selected_cats = None
            self.selected_objs = None
            self.update_img(local=False)

        def last_img(self, event=None):
            self.data.go_to_image(self.data.images.max - 1)
            self.set_locals()
            self.selected_cats = None
            self.selected_objs = None
            self.update_img(local=False)

        def bind_events(self):
            self.root.bind("<Left>", self.prev_img)
            self.root.bind("<Right>", self.next_img)
            self.root.bind("<k>", self.prev_img)
            self.root.bind("<j>", self.next_img)
            self.root.bind("<Home>", self.first_img)
            self.root.bind("<End>", self.last_img)
            self.root.bind("<Control-o>", self.load_annotations)
            self.root.bind("<Control-q>", self.exit)
            self.root.bind("<Control-s>", self.save_image)
            self.root.bind("<b>", self.toggle_bboxes)
            self.root.bind("<l>", self.toggle_labels)
            self.root.bind("<m>", self.toggle_masks)
            self.root.bind("<space>", self.toggle_all)
            self.objects_panel.category_box.bind("<<ListboxSelect>>", self.select_category)
            self.objects_panel.object_box.bind("<<ListboxSelect>>", self.select_object)
            self.image_panel.bind("<Button-1>", lambda e: self.image_panel.focus_set())


def main():
    args = parser.parse_args()

    if args.check:
        ann = args.annotations or str(DEFAULT_ANNOTATIONS)
        img_root = args.images or str(DEFAULT_DATA_ROOT)
        issues = check_dataset(ann, img_root)
        sys.exit(1 if issues > 0 else 0)

    if not HAS_TK:
        logging.error("tkinter not available. Use --check for validation only.")
        sys.exit(1)

    root = tk.Tk()
    root.title("COCO Dataset Viewer - Wound Detection")

    ann = args.annotations or str(DEFAULT_ANNOTATIONS)
    img_root = args.images or str(DEFAULT_DATA_ROOT)

    resolved = resolve_annotation_path(ann)
    if not resolved.exists():
        messagebox.showinfo(
            "Annotations not found",
            f"The annotations file was not found:\n{resolved}\n\n"
            f"Select the folder that contains «{resolved.name}», or cancel.",
        )
        folder = filedialog.askdirectory(
            parent=root,
            title=f"Select folder containing {resolved.name}",
            mustexist=True,
        )
        if not folder:
            sys.exit(0)
        candidate = Path(folder) / resolved.name
        if candidate.exists():
            ann = str(candidate.resolve())
        else:
            picked = filedialog.askopenfilename(
                parent=root,
                title="JSON not in that folder — select annotations file",
                initialdir=folder,
                filetypes=(("JSON files", "*.json"), ("all files", "*.*")),
            )
            if not picked:
                sys.exit(0)
            ann = picked
    else:
        ann = str(resolved)

    data = Data(img_root, ann)
    statusbar = StatusBar(root)
    sliders = SlidersBar(root)
    images_panel = ImagesPanel(root)
    objects_panel = ObjectsPanel(root)
    menu = Menu(root)
    image_panel = ImagePanel(root)
    Controller(data, root, image_panel, statusbar, menu, objects_panel, images_panel, sliders, annotations_path=ann)
    root.mainloop()


if __name__ == "__main__":
    main()
