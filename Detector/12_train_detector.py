"""
Step 12 — Train the pneumonia detector (YOLO, Phase 2)
=====================================================

Run in yolo-env (PyTorch). Needs yolo_data/ from step 11.

What is being trained
  YOLO11 "s" (small, ~9 M weights), pretrained on COCO (everyday photos: people, cars, ...).
  Same idea as DenseNet + ImageNet in Phase 1: the early layers already know edges and
  textures, we only teach it what a lung opacity looks like and where its edges are.
  One class: 0 = opacity.

How YOLO learns (per image)
  The image is cut into grids (64 x 64, 32 x 32, 16 x 16 cells at 512 px: fine grids for
  small opacities, coarse grids for large ones). Every cell predicts:
    - a box (centre, width, height)          -> "box loss": how far from the radiologist's box
    - how sure it is that this is opacity    -> "cls loss": opacity vs background
    - the box edges as a small distribution  -> "dfl loss": sharp, well-placed edges
  Background images (empty label files) only produce "nothing here" lessons, which is how
  the detector learns not to draw boxes on normal or Not-Normal lungs.

Settings and why
  imgsz 512        our PNGs are 512 px; larger would only upscale
  epochs 60, patience 15   stops early if validation mAP has not improved for 15 epochs
  batch 16         fits the 8 GB RTX 4060 at 512 px (use 8 if 'CUDA out of memory')
  fliplr 0.0       no left-right flip, as in the classifier (heart side). Flip is tested
                   later as a separate experiment (--fliplr 0.5)
  degrees 10, translate 0.06, scale 0.25    close to the classifier's augmentation
  hsv_h 0, hsv_s 0, hsv_v 0.3   X-rays are grey: hue / saturation changes do nothing;
                   brightness changes like a different exposure
  mosaic 1.0       YOLO's default: 4 training images tiled into one (more boxes per step,
                   boxes at more positions); switched off for the last 10 epochs so the
                   model finishes on normal, whole X-rays
  seed 42, deterministic

Validation during training uses ONLY the val split. The test split is never touched here.
The best epoch is chosen by YOLO's fitness = 0.1 * mAP@0.5 + 0.9 * mAP@0.5:0.95.

Output: <outputs>/detector_runs/<name>/
    weights/best.pt, weights/last.pt     the model (best = highest validation fitness)
    results.csv, results.png             loss and mAP per epoch
    our_config.json                      every setting used
    val_summary.txt                      final validation numbers of best.pt

Run (inside WSL, yolo-env active, from the code folder):
    python 12_train_detector.py --data "$HOME/AIML Project/Test 2/RSNA/outputs/yolo_data/data.yaml" --smoke
    python 12_train_detector.py --data "$HOME/AIML Project/Test 2/RSNA/outputs/yolo_data/data.yaml"
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="yolo_data/data.yaml from step 11")
    ap.add_argument("--model", default="yolo11s.pt", help="pretrained weights (downloaded on first use)")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fliplr", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0", help="'0' = first GPU, 'cpu' for testing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help="extra word in the run name")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 5 %% of the data, to test the set-up")
    args = ap.parse_args()

    stem = Path(args.model).stem
    name = f"{stem}_{args.imgsz}" + ("_flip" if args.fliplr > 0 else "") + ("_smoke" if args.smoke else "") \
        + (f"_{args.tag}" if args.tag else "") + datetime.now().strftime("_%Y%m%d-%H%M")
    project = (args.data.parent.parent / "detector_runs").resolve()     # <outputs>/detector_runs

    settings = dict(
        data=str(args.data), imgsz=args.imgsz, epochs=1 if args.smoke else args.epochs,
        patience=args.patience, batch=args.batch, workers=args.workers, device=args.device,
        seed=args.seed, deterministic=True, project=str(project), name=name, exist_ok=False,
        fraction=0.05 if args.smoke else 1.0,
        # augmentation
        fliplr=args.fliplr, flipud=0.0, degrees=10.0, translate=0.06, scale=0.25, shear=0.0,
        perspective=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, mosaic=1.0, mixup=0.0, close_mosaic=10,
        # optimiser: YOLO's defaults (SGD / AdamW chosen automatically, warm-up 3 epochs)
        optimizer="auto", cos_lr=False, amp=True,
        plots=True, val=True, verbose=True,
    )

    model = YOLO(args.model)
    model.train(**settings)
    run_dir = Path(model.trainer.save_dir)
    (run_dir / "our_config.json").write_text(json.dumps({**settings, "model": args.model}, indent=2))

    # final check of the best weights on the validation split (never test)
    best = YOLO(run_dir / "weights" / "best.pt")
    m = best.val(data=str(args.data), split="val", imgsz=args.imgsz, batch=args.batch,
                 device=args.device, plots=False, verbose=False, project=str(run_dir), name="val_best")
    lines = [f"run: {run_dir.name}",
             f"validation images: all val (Normal + Not-Normal + pneumonia)",
             f"precision  {m.box.mp:.3f}   (of the boxes drawn at YOLO's default cut-off, share that match a real box)",
             f"recall     {m.box.mr:.3f}   (share of real boxes found)",
             f"mAP@0.5    {m.box.map50:.3f}   (a box counts if it overlaps a real box by >= 50 %)",
             f"mAP@0.5:0.95 {m.box.map:.3f}  (stricter: averaged over overlap 50 % ... 95 %)"]
    (run_dir / "val_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nSaved in {run_dir}")


if __name__ == "__main__":
    main()
