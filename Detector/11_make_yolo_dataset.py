"""
Step 11 — Turn our split + boxes into a YOLO dataset (Phase 2, the detector)
===========================================================================

Run in yolo-env (PyTorch). Nothing here is trained; it only arranges files.

What YOLO wants
  datasets/
    images/train/<id>.png    images/val/...    images/test/...
    labels/train/<id>.txt    labels/val/...    labels/test/...
    data.yaml                 (where the folders are + the class names)
  One .txt per image, one line per box:     class  x_centre  y_centre  width  height
  all four numbers as a FRACTION of the image size (0 to 1), class 0 = "opacity".
  An image whose .txt is EMPTY is a "background" image: it teaches the detector where
  NOT to draw boxes. boxes.csv (step 2) already holds the YOLO numbers (yolo_xc ... yolo_h).

The rules this script follows
  1. SAME split as the classifier (the 'split' column of metadata.csv). If a classifier
     test image ended up in the detector's training set, the final system's test result
     would be leaked.
  2. Every image is a symbolic link to the PNG from step 1 (no copies; 0 extra disk).
     YOLO finds a label by swapping '/images/' for '/labels/' in the image path, so the
     link must live inside images/<split>/.
  3. Background images:
       train : Normal + Not-Normal, by default as many as there are pneumonia images
               (--bg-per-pos 1.0, half Normal, half Not-Normal). All of them would be
               ~3x more images per epoch; that can be tried later as an experiment.
       val / test : ALL images of every class, so the false-box rate is measured honestly.
     Use --no-not-normal to leave the Not-Normal images out of training (the comparison
     experiment).
  4. manifest.csv lists every image with its class, view, sex and age, for the
     per-group evaluation later (e.g. false boxes in the lower zones of female X-rays).

Checks
  - every pneumonia image gets >= 1 box line, every background image an empty file
  - every box lies inside the image
  - preview_train.png: 8 training images with the boxes drawn FROM THE LABEL FILES
    (if the boxes sit on the lungs as in step 2's pictures, the format is right)

Run (inside WSL, yolo-env active, from the code folder):
    python 11_make_yolo_dataset.py --csv "$HOME/AIML Project/Test 2/RSNA/outputs/metadata.csv"
Output: <outputs>/yolo_data/  (data.yaml, images/, labels/, manifest.csv, preview_train.png)
"""

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SEED = 42


def resolve(p, base):
    p = Path(p)
    return p if p.is_absolute() or p.exists() else (base / p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="metadata.csv from step 1")
    ap.add_argument("--boxes", type=Path, default=None, help="boxes.csv (default: next to metadata.csv)")
    ap.add_argument("--out", type=Path, default=None, help="default: <csv folder>/yolo_data")
    ap.add_argument("--bg-per-pos", type=float, default=1.0,
                    help="training background images per pneumonia image (0 = none, -1 = all)")
    ap.add_argument("--no-not-normal", action="store_true", help="training background = Normal only")
    ap.add_argument("--copy", action="store_true", help="copy images instead of linking (e.g. on a Windows drive)")
    args = ap.parse_args()

    out = args.out or args.csv.parent / "yolo_data"
    if out.exists():
        shutil.rmtree(out)                                   # always rebuild from scratch
    meta = pd.read_csv(args.csv)
    boxes = pd.read_csv(args.boxes or args.csv.parent / "boxes.csv")
    base = args.csv.parent
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------ 1. choose images
    parts = []
    for split in ("train", "val", "test"):
        m = meta[meta["split"] == split]
        pos = m[m["class"] == "opacity"]
        if split != "train":
            parts.append(m)                                  # val / test: everything
            continue
        bg_classes = ["normal"] + ([] if args.no_not_normal else ["not_normal"])
        bg = m[m["class"].isin(bg_classes)]
        if args.bg_per_pos >= 0:
            n_bg = min(len(bg), int(round(args.bg_per_pos * len(pos))))
            per = n_bg // len(bg_classes)                    # equal share per class
            bg = pd.concat([g.sample(min(len(g), per), random_state=SEED) for _, g in bg.groupby("class")])
        parts.append(pd.concat([pos, bg]))
    sel = pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------------ 2. write files
    by_pid = dict(tuple(boxes.groupby("patientId")))
    problems = []
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
    rows = []
    # 'class' is a Python keyword, so the rows are read as plain dictionaries: r["class"]
    for r in sel.to_dict("records"):
        src = resolve(r["png_path"], base).resolve()
        if not src.exists():
            problems.append(f"missing image {src}")
            continue
        dst = out / "images" / r["split"] / f"{r['patientId']}.png"
        if args.copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)
        lines = []
        if r["class"] == "opacity":
            b = by_pid.get(r["patientId"])
            if b is None or len(b) == 0:
                problems.append(f"opacity image without boxes: {r['patientId']}")
            else:
                for q in b.itertuples(index=False):
                    xc, yc, w, h = q.yolo_xc, q.yolo_yc, q.yolo_w, q.yolo_h
                    eps = 1e-6
                    if xc - w / 2 < -eps or xc + w / 2 > 1 + eps or yc - h / 2 < -eps or yc + h / 2 > 1 + eps:
                        problems.append(f"box outside image: {r['patientId']}")
                    lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        (out / "labels" / r["split"] / f"{r['patientId']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        rows.append({"patientId": r["patientId"], "split": r["split"], "class": r["class"],
                     "view": r["view"], "sex": r["sex"], "age": r["age"], "n_boxes": len(lines),
                     "image": str(dst)})
    man = pd.DataFrame(rows)
    man.to_csv(out / "manifest.csv", index=False)

    (out / "data.yaml").write_text(
        f"# written by 11_make_yolo_dataset.py - same split as the classifier\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n  0: opacity\n")

    # ------------------------------------------------------------------ 3. summary + checks
    print("=" * 72)
    print(f"YOLO dataset written to {out}")
    print("=" * 72)
    tab = man.pivot_table(index="split", columns="class", values="patientId", aggfunc="count", fill_value=0)
    tab["boxes"] = man.groupby("split")["n_boxes"].sum()
    tab["background %"] = (100 * (man["n_boxes"] == 0).groupby(man["split"]).mean()).round(1)
    print(tab.reindex(["train", "val", "test"]).to_string())
    print(f"\nTraining background: {'Normal only' if args.no_not_normal else 'Normal + Not-Normal'}, "
          f"bg-per-pos = {args.bg_per_pos}")
    leaked = set(man.loc[man.split == "train", "patientId"]) & set(meta.loc[meta.split != "train", "patientId"])
    print(f"Split check: {len(leaked)} training images belong to the classifier's val/test (must be 0)")
    print(f"Problems: {len(problems)}" + ("" if not problems else "  e.g. " + "; ".join(problems[:3])))

    # preview drawn from the label files themselves
    tr = man[(man.split == "train") & (man.n_boxes > 0)].sample(min(8, int((man.n_boxes > 0).sum())), random_state=SEED)
    tiles = []
    for t in tr.itertuples(index=False):
        img = cv2.imread(t.image)
        H, W = img.shape[:2]
        for line in (out / "labels" / "train" / f"{t.patientId}.txt").read_text().split("\n"):
            if not line.strip():
                continue
            _, xc, yc, w, h = map(float, line.split())
            cv2.rectangle(img, (int((xc - w / 2) * W), int((yc - h / 2) * H)),
                          (int((xc + w / 2) * W), int((yc + h / 2) * H)), (0, 0, 255), 2)
        cv2.putText(img, f"{t.view} {t.sex} {t.age}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(cv2.resize(img, (256, 256)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(out / "preview_train.png"), grid)
    print(f"Preview (boxes drawn from the label files): {out / 'preview_train.png'}")


if __name__ == "__main__":
    main()
