"""
Step 1 — RSNA data audit + split + PNG conversion
==================================================

What this script does, in order:
  1. Loads the two label CSVs and joins them  -> one row per patient
  2. Reads every DICOM *header* (view, age, sex, pixel spacing)
  3. Splits patients 70/15/15, stratified by class x view, with a fixed seed
  4. Prints audit tables (class x view, class x sex, age, split check) and saves them to audit.txt
  5. Converts every DICOM to a 512x512 greyscale PNG (skips files already converted)
  6. Saves metadata.csv  -> the single source of truth for every later step

Run (inside WSL):
    pip install pandas pydicom pillow opencv-python-headless scikit-learn tqdm
    python 01_audit_split.py

Default folders (change with --raw / --png / --out if yours differ):
    ~/rsna/raw      the copied RSNA download (CSVs + stage_2_train_images/)
    ~/rsna/png      where the PNGs will be written
    ~/rsna/outputs  where metadata.csv and audit.txt will be written
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SEED = 42          # fixed seed -> the split comes out identical every run
PNG_SIZE = 512     # 1024 -> 512 keeps enough detail to train at 224 now, 384/512 later


# ----------------------------------------------------------------------------
# 1. Labels: join the two CSVs into one row per patient
# ----------------------------------------------------------------------------
def load_labels(raw: Path) -> pd.DataFrame:
    labels = pd.read_csv(raw / "stage_2_train_labels.csv")          # one row per BOX
    classes = pd.read_csv(raw / "stage_2_detailed_class_info.csv")  # same rows, 3-way class

    # Number of boxes per patient (0 for patients without opacity)
    n_boxes = (labels.assign(has_box=labels["x"].notna())
                     .groupby("patientId")["has_box"].sum()
                     .astype(int).rename("n_boxes"))

    # Target is identical on every row of a patient, so take the max
    target = labels.groupby("patientId")["Target"].max().rename("target")

    cls = classes.drop_duplicates("patientId").set_index("patientId")["class"]

    df = pd.concat([cls, target, n_boxes], axis=1).reset_index()
    df = df.rename(columns={"index": "patientId"})

    # Short class names are easier to type later
    df["class"] = df["class"].map({
        "Lung Opacity": "opacity",
        "Normal": "normal",
        "No Lung Opacity / Not Normal": "not_normal",
    })
    return df


# ----------------------------------------------------------------------------
# 2. DICOM headers (stop_before_pixels=True reads only the "form", not the image -> fast)
# ----------------------------------------------------------------------------
def read_header(dcm_path: str) -> dict:
    d = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    spacing = d.get("PixelSpacing", [None])[0]
    age = d.get("PatientAge", None)
    return {
        "view": d.get("ViewPosition", None),
        "age": int(age) if age not in (None, "") else None,
        "sex": d.get("PatientSex", None),
        "pixel_spacing": round(float(spacing), 4) if spacing is not None else None,
    }


# ----------------------------------------------------------------------------
# 5. Conversion: DICOM pixels -> resized greyscale PNG
# ----------------------------------------------------------------------------
def convert_one(args) -> str:
    dcm_path, png_path = args
    if os.path.exists(png_path):          # already done -> lets you re-run safely
        return "skipped"
    img = pydicom.dcmread(dcm_path).pixel_array      # 1024x1024, uint8 in RSNA
    if img.dtype != np.uint8:                        # safety net for other datasets
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.resize(img, (PNG_SIZE, PNG_SIZE), interpolation=cv2.INTER_AREA)
    cv2.imwrite(png_path, img)
    return "written"


# ----------------------------------------------------------------------------
# 4. Audit tables
# ----------------------------------------------------------------------------
def audit(df: pd.DataFrame) -> str:
    out = []
    p = out.append
    p(f"Patients: {len(df)}\n")
    p("Class counts:\n" + df["class"].value_counts().to_string() + "\n")

    p("Class x view (counts):\n" + pd.crosstab(df["class"], df["view"], margins=True).to_string() + "\n")
    p("Class x view (share of each view, %)  <- the AP shortcut check:\n"
      + (pd.crosstab(df["class"], df["view"], normalize="columns") * 100).round(1).to_string() + "\n")

    p("Class x sex (share of each sex, %):\n"
      + (pd.crosstab(df["class"], df["sex"], normalize="columns") * 100).round(1).to_string() + "\n")

    p("Age by class:\n" + df.groupby("class")["age"].describe().round(1).to_string() + "\n")
    bad = df[(df["age"] > 100) | (df["age"] < 0) | df["age"].isna()]
    p(f"Suspicious ages (>100, <0 or missing): {len(bad)}")
    if len(bad):
        p(bad[["patientId", "class", "age"]].to_string(index=False))
    p("")

    p("Pixel spacing values (scanner hint) x view:\n"
      + pd.crosstab(df["pixel_spacing"], df["view"]).to_string() + "\n")

    p("Split check — class x view share per split (%), should look the same in all three:\n"
      + (pd.crosstab([df["split"]], [df["class"], df["view"]], normalize="index") * 100)
        .round(1).to_string() + "\n")
    p("Split sizes:\n" + df["split"].value_counts().to_string() + "\n")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# 3. Stratified 70/15/15 split
# ----------------------------------------------------------------------------
def split(df: pd.DataFrame) -> pd.DataFrame:
    strata = df["class"] + "_" + df["view"].fillna("NA")
    train_ids, rest_ids = train_test_split(
        df["patientId"], test_size=0.30, stratify=strata, random_state=SEED)
    rest = df.set_index("patientId").loc[rest_ids]
    rest_strata = rest["class"] + "_" + rest["view"].fillna("NA")
    val_ids, test_ids = train_test_split(
        rest.index, test_size=0.50, stratify=rest_strata, random_state=SEED)

    df = df.copy()
    df["split"] = "train"
    df.loc[df["patientId"].isin(val_ids), "split"] = "val"
    df.loc[df["patientId"].isin(test_ids), "split"] = "test"
    return df


def main():
    home = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=home / "rsna" / "raw")
    ap.add_argument("--png", type=Path, default=home / "rsna" / "png")
    ap.add_argument("--out", type=Path, default=home / "rsna" / "outputs")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--skip-convert", action="store_true",
                    help="only audit + split, no PNG conversion")
    args = ap.parse_args()

    args.png.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    img_dir = args.raw / "stage_2_train_images"

    # 1. labels
    df = load_labels(args.raw)
    print(f"[1/6] Labels loaded: {len(df)} patients")

    # Keep only patients whose DICOM file actually exists (catches incomplete copies)
    df["dcm_path"] = df["patientId"].map(lambda pid: str(img_dir / f"{pid}.dcm"))
    exists = df["dcm_path"].map(os.path.exists)
    if not exists.all():
        print(f"      WARNING: {(~exists).sum()} DICOM files are missing — they will be skipped")
    df = df[exists].reset_index(drop=True)

    # 2. headers (parallel)
    with ProcessPoolExecutor(args.workers) as ex:
        headers = list(tqdm(ex.map(read_header, df["dcm_path"], chunksize=64),
                            total=len(df), desc="[2/6] Reading headers"))
    df = pd.concat([df, pd.DataFrame(headers)], axis=1)

    # 3. split (done before the audit so the audit can also check the split)
    df = split(df)
    print("[3/6] Split done")

    # 4. audit
    report = audit(df)
    print("\n[4/6] AUDIT\n" + "=" * 60 + "\n" + report)
    (args.out / "audit.txt").write_text(report)

    # 5. convert
    df["png_path"] = df["patientId"].map(lambda pid: str(args.png / f"{pid}.png"))
    if not args.skip_convert:
        jobs = list(zip(df["dcm_path"], df["png_path"]))
        with ProcessPoolExecutor(args.workers) as ex:
            results = list(tqdm(ex.map(convert_one, jobs, chunksize=32),
                                total=len(jobs), desc="[5/6] Converting to PNG"))
        print(f"      written: {results.count('written')}, skipped (already existed): {results.count('skipped')}")

    # 6. save
    cols = ["patientId", "class", "target", "view", "age", "sex",
            "pixel_spacing", "n_boxes", "split", "png_path"]
    df[cols].to_csv(args.out / "metadata.csv", index=False)
    print(f"[6/6] Saved {args.out / 'metadata.csv'} and {args.out / 'audit.txt'}")


if __name__ == "__main__":
    main()
