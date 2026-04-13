"""
tile.py  –  2×2 overlapping image tiling for the BCCD training set.

Why this helps
--------------
Platelets average ~41×41 px in a 640×480 frame.  At 960 px processing
resolution (used by DeformableDetrImageProcessor) the model's coarsest
feature map has stride 32, so a 41 px object occupies only ~1.3 cells —
right on the edge of detectability.

Tiling into four 380×300 px patches doubles the effective object size:
a 41 px platelet now covers ~2.6 cells in the coarsest feature map and
is clearly visible in the finest multi-scale attention layers.

Usage
-----
Run once after preprocess.py to generate the tiled training set:

    uv run python tile.py

This writes:
    blood_cell_data/train_tiled/      (1164 tile images)
    blood_cell_data/train_tiled.json  (COCO annotations in tile coordinates)

Then train with tiling enabled:

    uv run python train.py --tiled

Tile geometry (640×480 source images)
--------------------------------------
    Overlap  : 120 px (horizontal and vertical)
    Tile size : 380×300 px
    Strides  : 260 px (H), 180 px (V)
    Tiles per image: 4  (2 cols × 2 rows)

    col 0: x ∈ [  0, 380]   col 1: x ∈ [260, 640]
    row 0: y ∈ [  0, 300]   row 1: y ∈ [180, 480]
"""

import os
import json
import math
from pathlib import Path

from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Tile geometry
# ---------------------------------------------------------------------------
IMG_W, IMG_H = 640, 480
N_COLS, N_ROWS = 2, 2
OVERLAP = 120          # px shared between neighbouring tiles
TILE_W = math.ceil(IMG_W / N_COLS) + OVERLAP // 2   # 380 px
TILE_H = math.ceil(IMG_H / N_ROWS) + OVERLAP // 2   # 300 px
STRIDE_X = IMG_W - TILE_W                           # 260 px
STRIDE_Y = IMG_H - TILE_H                           # 180 px

# A bounding box must have at least this fraction of its area inside a tile
# to be included in that tile's annotation list.
MIN_VISIBILITY = 0.30


def get_tiles(img_w=IMG_W, img_h=IMG_H):
    """Return list of (x1, y1, x2, y2) tile boxes for a single image."""
    tile_w = math.ceil(img_w / N_COLS) + OVERLAP // 2
    tile_h = math.ceil(img_h / N_ROWS) + OVERLAP // 2
    stride_x = img_w - tile_w
    stride_y = img_h - tile_h

    tiles = []
    for row in range(N_ROWS):
        for col in range(N_COLS):
            x1 = col * stride_x
            y1 = row * stride_y
            x2 = min(x1 + tile_w, img_w)
            y2 = min(y1 + tile_h, img_h)
            tiles.append((x1, y1, x2, y2))
    return tiles


def clip_box_to_tile(bbox_coco, tx1, ty1, tx2, ty2, min_vis=MIN_VISIBILITY):
    """
    Intersect a COCO [x, y, w, h] box with a tile and return the clipped box
    in tile-local coordinates, or None if visibility < min_vis.
    """
    bx, by, bw, bh = bbox_coco
    orig_area = bw * bh
    if orig_area <= 0:
        return None

    # Intersection
    ix1 = max(bx, tx1)
    iy1 = max(by, ty1)
    ix2 = min(bx + bw, tx2)
    iy2 = min(by + bh, ty2)

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    if inter_area / orig_area < min_vis:
        return None

    return [ix1 - tx1, iy1 - ty1, ix2 - ix1, iy2 - iy1]   # [x, y, w, h] local


# ---------------------------------------------------------------------------
# Main tiling function
# ---------------------------------------------------------------------------
def tile_coco_split(img_dir, ann_file, out_img_dir, out_ann_file):
    """
    Tile every image in a COCO split into N_COLS × N_ROWS overlapping patches.

    Parameters
    ----------
    img_dir      : folder containing source images
    ann_file     : COCO JSON for this split
    out_img_dir  : destination folder for tile images
    out_ann_file : destination COCO JSON with tile-local coordinates
    """
    os.makedirs(out_img_dir, exist_ok=True)

    with open(ann_file) as f:
        coco = json.load(f)

    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    new_images, new_annotations = [], []
    new_img_id = new_ann_id = 0

    for img_info in tqdm(coco["images"], desc=f"Tiling → {out_img_dir}"):
        src_path = os.path.join(img_dir, img_info["file_name"])
        if not os.path.exists(src_path):
            continue

        image = Image.open(src_path).convert("RGB")
        img_w, img_h = image.size
        anns = ann_by_img.get(img_info["id"], [])
        tiles = get_tiles(img_w, img_h)

        for t_idx, (tx1, ty1, tx2, ty2) in enumerate(tiles):
            tile_img = image.crop((tx1, ty1, tx2, ty2))
            tile_w, tile_h = tile_img.size

            stem = Path(img_info["file_name"]).stem
            tile_fname = f"{stem}_t{t_idx}.jpg"
            tile_img.save(os.path.join(out_img_dir, tile_fname), quality=95)

            new_img_id += 1
            new_images.append({
                "id": new_img_id,
                "file_name": tile_fname,
                "width": tile_w,
                "height": tile_h,
            })

            for ann in anns:
                clipped = clip_box_to_tile(ann["bbox"], tx1, ty1, tx2, ty2)
                if clipped is None:
                    continue
                new_ann_id += 1
                new_annotations.append({
                    "id": new_ann_id,
                    "image_id": new_img_id,
                    "category_id": ann["category_id"],
                    "bbox": clipped,
                    "area": clipped[2] * clipped[3],
                    "iscrowd": ann.get("iscrowd", 0),
                })

    out = {"images": new_images, "annotations": new_annotations,
           "categories": coco["categories"]}
    with open(out_ann_file, "w") as f:
        json.dump(out, f)

    orig_anns = len(coco["annotations"])
    new_anns = len(new_annotations)
    print(f"\n  Images : {len(coco['images'])} → {len(new_images)} tiles")
    print(f"  Annotations : {orig_anns} → {new_anns}  "
          f"({new_anns / orig_anns:.1f}× more training signal)")
    print(f"  Saved  : {out_img_dir}  |  {out_ann_file}")


# ---------------------------------------------------------------------------
# Per-class annotation count helper (for verification)
# ---------------------------------------------------------------------------
def print_class_stats(ann_file, label=""):
    with open(ann_file) as f:
        d = json.load(f)
    cats = {c["id"]: c["name"] for c in d["categories"]}
    from collections import Counter
    counts = Counter(a["category_id"] for a in d["annotations"])
    print(f"\n{'Class distribution':30s}  [{label}]")
    for cid, name in cats.items():
        print(f"  {name:12s}: {counts[cid]:5d}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("BCCD Image Tiling  (2×2 grid, 120 px overlap)")
    print(f"Tile size: {TILE_W}×{TILE_H} px  |  Stride: {STRIDE_X}×{STRIDE_Y} px")
    print("=" * 60)

    tile_coco_split(
        img_dir="blood_cell_data/train",
        ann_file="blood_cell_data/train.json",
        out_img_dir="blood_cell_data/train_tiled",
        out_ann_file="blood_cell_data/train_tiled.json",
    )

    print_class_stats("blood_cell_data/train.json",      label="original")
    print_class_stats("blood_cell_data/train_tiled.json", label="tiled")

    print("\nDone. Run training with:  uv run python train.py --tiled")
