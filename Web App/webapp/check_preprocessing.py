"""
check_preprocessing.py — where do the website's logits differ from step 10, and does it matter?
==============================================================================================

Run in ai-env (needs TensorFlow + onnxruntime, both installed for step 18), from the webapp folder:
    python check_preprocessing.py --outputs "$HOME/AIML Project/Test 2/RSNA/outputs" --n 200

For n validation X-rays, every member and every TTA version, it compares three logits:
    A  step 10's cache            TensorFlow image preparation + mixed-precision (float16) model
    B  ONNX + TensorFlow prep     TensorFlow image preparation + float32 ONNX model
    C  ONNX + website prep        the website's OpenCV/numpy preparation + float32 ONNX model
  B - A = the precision effect (float16 vs float32), nothing to do with the website code
  C - B = the website's image preparation (OpenCV 'area' resize vs TensorFlow's)
Then it compares the FINAL probabilities and decisions (all members and TTA averaged, calibrated)
for A and C, which is what a user of the website sees.
"""

import tensorflow as tf

for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import calibration_utils as cu                     # noqa: E402
from data_pipeline import _load_png                # noqa: E402
import pipeline as P                               # noqa: E402

F = importlib.import_module("10_final_model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    mdir = a.outputs / "webapp_models"
    man = json.loads((mdir / "manifest.json").read_text())
    cal = cu.load(mdir / "calibration_v2.json")
    cache = np.load(a.outputs / "final_model" / "cache_val.npz", allow_pickle=True)
    ids = list(cache["ids"])
    meta = pd.read_csv(a.outputs / "metadata.csv")
    val = meta[(meta.split == "val") & meta["class"].isin(["normal", "opacity"])].head(a.n)
    clf_np = P.Classifier(mdir, man)                              # the website's preparation
    sessions = [ort.InferenceSession(str(mdir / m["onnx"]), providers=["CPUExecutionProvider"])
                for m in man["classifier"]["members"]]
    mean, std = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)

    BA, CB, CA, rows = [], [], [], []
    for r in val.itertuples(index=False):
        p = Path(r.png_path); p = p if p.is_absolute() else a.outputs / p
        A = np.stack([cache[m["source_run"]][ids.index(r.patientId)] for m in man["classifier"]["members"]])
        B = []
        for sess, m in zip(sessions, man["classifier"]["members"]):
            x = tf.cast(_load_png(str(p), m["img_size"]), tf.float32) / 255.0
            x = tf.image.grayscale_to_rgb(x) * 255.0
            batch = np.stack([((F.affine(x[None], ang, z).numpy()[0] / 255.0 - mean) / std) for _, ang, z in man["classifier"]["tta"]])
            B.append(sess.run(None, {sess.get_inputs()[0].name: batch.astype(np.float32)})[0].ravel())
        B = np.array(B)
        img, _ = P.read_image(p.read_bytes(), p.name)
        C, _ = clf_np(img)
        BA.append(np.abs(B - A).max()); CB.append(np.abs(C - B).max()); CA.append(np.abs(C - A).max())
        v = r.view if r.view in ("AP", "PA") else "unknown"
        pa, pc = (float(cu.probability(cal, [z.mean()], [v])[0]) for z in (A, C))
        da, dc = (int(cu.decide(cal, [z.mean()], [v])[0]) for z in (A, C))
        rows.append((r.patientId, pa, pc, da, dc))

    q = lambda x: f"median {np.median(x):.4f}, 95th pct {np.percentile(x, 95):.4f}, max {np.max(x):.4f}"
    print(f"\n{len(val)} validation X-rays, largest logit difference per X-ray (over members and TTA versions):")
    print(f"  B-A  precision (float16 vs float32)          : {q(BA)}")
    print(f"  C-B  website image preparation               : {q(CB)}")
    print(f"  C-A  website vs step 10 in total             : {q(CA)}")
    R = pd.DataFrame(rows, columns=["id", "p_step10", "p_web", "d_step10", "d_web"])
    dp = (R.p_web - R.p_step10).abs()
    print(f"\nFinal calibrated probability, website vs step 10: median diff {dp.median():.4f}, max {dp.max():.4f}")
    print(f"Decisions identical: {(R.d_web == R.d_step10).sum()}/{len(R)}")
    for x in R[R.d_web != R.d_step10].itertuples():
        print(f"  changed: {x.id[:8]} probability {x.p_step10:.3f} -> {x.p_web:.3f}")


if __name__ == "__main__":
    main()
