"""
Step 2 — Bounding boxes into one clean table (boxes.csv)
=========================================================

Why this step exists
  metadata.csv (from step 1) has ONE row per patient and only the NUMBER of boxes.
  The detector (step 6) and the zone/extent evaluation (step 8) need the boxes themselves.
  This script writes them to boxes.csv, one row per box, with the same split as metadata.csv,
  so every later step reads boxes from one place and can never use a different split.

What it does, in order:
  1. Reads stage_2_train_labels.csv and keeps only rows that really are boxes (Target = 1)
  2. Adds the split / view / class from metadata.csv
  3. Adds convenience columns: box in 512-px PNG coordinates, box in YOLO format,
     box area as % of the image, and which half of the image the box centre is in
  4. Runs sanity checks (every opacity patient has >= 1 box, no box leaves the image, ...)
  5. Saves boxes.csv next to metadata.csv and prints a short summary

Run (inside WSL):
    python 02_make_boxes_csv.py --raw ~/rsna/raw --meta ~/rsna/outputs/metadata.csv

Box format reminder
  RSNA gives x, y = TOP-LEFT corner, width, height, in pixels of the ORIGINAL 1024 x 1024 image.
  Image x grows to the right, image y grows DOWNWARDS.
  The patient's RIGHT lung is on the LEFT of the image (x < 512).
"""

import argparse
from pathlib import Path

import pandas as pd

ORIG = 1024   # RSNA DICOMs are 1024 x 1024
PNG = 512     # size step 1 saved the PNGs at


def main():
    home = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=home / "rsna" / "raw",
                    help="folder with stage_2_train_labels.csv")
    ap.add_argument("--meta", type=Path, default=home / "rsna" / "outputs" / "metadata.csv",
                    help="metadata.csv from step 1")
    ap.add_argument("--out", type=Path, default=None,
                    help="output file (default: boxes.csv next to metadata.csv)")
    args = ap.parse_args()
    out = args.out or args.meta.parent / "boxes.csv"

    # ------------------------------------------------------------------ 1. boxes only
    labels = pd.read_csv(args.raw / "stage_2_train_labels.csv")
    boxes = labels[labels["Target"] == 1].copy()
    boxes = boxes.rename(columns={"width": "w", "height": "h"})
    boxes[["x", "y", "w", "h"]] = boxes[["x", "y", "w", "h"]].astype(float)
    boxes["box_idx"] = boxes.groupby("patientId").cumcount()   # 0, 1, 2, 3 within a patient
    print(f"[1/5] {len(boxes)} boxes on {boxes['patientId'].nunique()} patients")

    # ------------------------------------------------------------------ 2. join split
    meta = pd.read_csv(args.meta)
    keep = [c for c in ("patientId", "split", "view", "class", "png_path") if c in meta.columns]
    boxes = boxes.merge(meta[keep], on="patientId", how="left")
    missing = boxes["split"].isna().sum()
    if missing:
        print(f"      WARNING: {missing} boxes belong to patients not in metadata.csv "
              f"(missing DICOM?) - they are dropped")
        boxes = boxes[boxes["split"].notna()]
    print("[2/5] Split added")

    # ------------------------------------------------------------------ 3. extra columns
    s = PNG / ORIG
    for c in ("x", "y", "w", "h"):
        boxes[f"{c}_png"] = (boxes[c] * s).round(1)
    # YOLO: centre x, centre y, width, height, all divided by image size (0..1)
    boxes["yolo_xc"] = ((boxes["x"] + boxes["w"] / 2) / ORIG).round(6)
    boxes["yolo_yc"] = ((boxes["y"] + boxes["h"] / 2) / ORIG).round(6)
    boxes["yolo_w"] = (boxes["w"] / ORIG).round(6)
    boxes["yolo_h"] = (boxes["h"] / ORIG).round(6)
    boxes["area_pct_image"] = (100 * boxes["w"] * boxes["h"] / ORIG ** 2).round(2)
    # rough side, from the box centre. Proper lung zones come later (step 8, lung mask).
    centre_x = boxes["x"] + boxes["w"] / 2
    boxes["image_half"] = (centre_x < ORIG / 2).map({True: "image_left", False: "image_right"})
    boxes["patient_side"] = boxes["image_half"].map({"image_left": "right", "image_right": "left"})
    print("[3/5] PNG / YOLO / area / side columns added")

    # ------------------------------------------------------------------ 4. sanity checks
    problems = []
    opacity_ids = set(meta.loc[meta["class"] == "opacity", "patientId"])
    with_box = set(boxes["patientId"])
    if opacity_ids - with_box:
        problems.append(f"{len(opacity_ids - with_box)} opacity patients have no box")
    if with_box - opacity_ids:
        problems.append(f"{len(with_box - opacity_ids)} patients have boxes but are not 'opacity'")
    outside = boxes[(boxes["x"] < 0) | (boxes["y"] < 0) |
                    (boxes["x"] + boxes["w"] > ORIG) | (boxes["y"] + boxes["h"] > ORIG)]
    if len(outside):
        problems.append(f"{len(outside)} boxes extend outside the 1024 x 1024 image")
    if "n_boxes" in meta.columns:
        per_pid = boxes.groupby("patientId").size()
        n_meta = meta.set_index("patientId").loc[per_pid.index, "n_boxes"]
        if (per_pid != n_meta).any():
            problems.append("box count differs from metadata.csv n_boxes for some patients")
    print("[4/5] Sanity checks: " + ("all passed" if not problems else "PROBLEMS FOUND"))
    for p in problems:
        print("      - " + p)

    # ------------------------------------------------------------------ 5. save + summary
    cols = ["patientId", "box_idx", "split", "view", "x", "y", "w", "h",
            "x_png", "y_png", "w_png", "h_png",
            "yolo_xc", "yolo_yc", "yolo_w", "yolo_h",
            "area_pct_image", "image_half", "patient_side"]
    if "png_path" in boxes.columns:
        cols.append("png_path")
    boxes[cols].to_csv(out, index=False)
    print(f"[5/5] Saved {out}\n")

    print("Boxes per split:")
    print(boxes.groupby("split").agg(patients=("patientId", "nunique"),
                                     boxes=("patientId", "size")).to_string(), "\n")
    print("Boxes per patient:")
    print(boxes.groupby("patientId").size().value_counts().sort_index()
          .rename_axis("boxes").rename("patients").to_string(), "\n")
    bil = boxes.groupby("patientId")["patient_side"].nunique()
    print(f"Patients with boxes on both sides (bilateral): {(bil == 2).sum()} of {len(bil)} "
          f"({100 * (bil == 2).mean():.1f} %)")
    print("Box size in original pixels (median):  "
          f"w {boxes['w'].median():.0f}, h {boxes['h'].median():.0f}; "
          f"area {boxes['area_pct_image'].median():.1f} % of the image")


if __name__ == "__main__":
    main()
