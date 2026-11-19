"""
Experiment A2 — Stress test on the Not-Normal images (evaluation only, no training)
===================================================================================

The question
  The classifier was trained and tuned on Normal vs Lung Opacity only. In real use it will
  also see X-rays that are abnormal for OTHER reasons (RSNA class "No Lung Opacity / Not
  Normal": effusions, enlarged heart, nodules, tubes and lines, old scarring, ...).
  How often does the model call those "pneumonia"?  That number is the false-alarm rate we
  should expect in practice, and it decides how much Experiment A3 (re-training with
  Not-Normal images as negatives) is needed.

What it does
  1. Loads the chosen run (best.keras + calibration_v2.json from steps 3 and 4b).
  2. Sanity check: re-predicts 64 validation images that already have a logit in
     val_predictions.csv and checks the logits match. If they do not, the images are being
     prepared differently from training and every later number would be wrong.
  3. Predicts every Not-Normal image in the validation split (about 1,770 images).
     Same calibration, same decision rule as the main model — nothing is re-tuned.
  4. Reports, overall and within AP / PA:
       a. share of each class flagged as pneumonia at the chosen threshold
          (for Not-Normal and Normal this is the false-alarm rate)
       b. which confidence band each class falls into
       c. AUROC of Opacity vs Normal, Opacity vs Not-Normal, Opacity vs both
       d. the "real-world mix": all three classes together, at their natural proportions
  5. Saves a gallery of the 16 Not-Normal images the model is MOST sure are pneumonia,
     so you can look at what fools it.

The test split is still not touched.

Output: <run folder>/not_normal_check/
    not_normal_report.txt        everything printed
    val_all_classes_predictions.csv   one row per validation image, all three classes
    top_false_alarms.png         gallery of the most confident Not-Normal false alarms

Run (inside WSL, from the code folder):
    python 06_not_normal_check.py \\
        --run "$HOME/AIML Project/Test 2/RSNA/outputs/runs/densenet121_384_20261025-1548" \\
        --csv "$HOME/AIML Project/Test 2/RSNA/outputs/metadata.csv"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

for _gpu in tf.config.list_physical_devices("GPU"):        # before anything touches the GPU
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

from sklearn.metrics import roc_auc_score

import calibration_utils as cu
from classifier_utils import logit_model, preprocess_fn
from data_pipeline import load_metadata, make_dataset

SEED = 42
CLASSES = ("normal", "not_normal", "opacity")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def wilson(k, n, z=1.96):
    """95 % confidence interval for a proportion k/n (better than +/- 2 SE for small n)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def predict_logits(lm, df, split, backbone, img_size, batch_size):
    prep = preprocess_fn(backbone)
    ds = make_dataset(df, split, img_size, batch_size, training=False, cache=False)
    ds = ds.map(lambda x, y: (prep(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return lm.predict(ds, verbose=1).ravel().astype("float64")


def auc_pair(d, neg_classes):
    """AUROC of Opacity (positive) vs the given negative classes: overall and within view."""
    m = d["class"].isin(("opacity",) + tuple(neg_classes))
    s = d[m]
    out = {"all": roc_auc_score(s["target"], s["prob"])}
    for v in ("AP", "PA"):
        sv = s[s["view"] == v]
        out[v] = roc_auc_score(sv["target"], sv["prob"]) if sv["target"].nunique() == 2 else np.nan
    out["within_view"] = np.nanmean([out["AP"], out["PA"]])
    return out


def gallery(rows, out_png, tile=256, cols=4):
    import cv2
    tiles = []
    for _, r in rows.iterrows():
        img = cv2.imread(str(r["png_path"]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img, (0, 0), (tile, 22), (0, 0, 0), -1)
        cv2.putText(img, f"p={r['prob']:.2f} {r['view']} {r['patientId'][:8]}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(img)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(out_png), grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run folder (needs best.keras + calibration_v2.json)")
    ap.add_argument("--csv", type=Path, required=True, help="metadata.csv from step 1")
    ap.add_argument("--split", default="val", choices=["val"], help="the test split stays untouched")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    cfg = json.loads((args.run / "config.json").read_text())
    cal = cu.load(args.run / "calibration_v2.json")          # per-view Platt + decision policy (step 4b)
    backbone, img_size = cfg["backbone"], int(cfg["img_size"])
    rule = f"policy {cal['decision']['policy']}, threshold(s) {cal['decision']['thresholds']}"
    out_dir = args.run / "not_normal_check"
    out_dir.mkdir(exist_ok=True)

    lines = []
    def say(s=""):
        print(s); lines.append(s)

    say(f"Run: {args.run.name} | {backbone} {img_size}px | {rule}")

    # ------------------------------------------------------------------ data
    meta = load_metadata(args.csv, include_not_normal=True)      # target: opacity=1, others=0
    meta = meta[meta["split"] == args.split].reset_index(drop=True)
    old = pd.read_csv(args.run / "val_predictions.csv")         # from step 3
    # A run trained WITH Not-Normal images (experiment A3) already predicted them in step 3:
    # then nothing new is predicted, otherwise those images would be counted twice.
    already = bool((old["class"] == "not_normal").any())
    nn = meta[meta["class"] == "not_normal"].reset_index(drop=True)
    if already:
        nn = nn.iloc[0:0]
        say(f"Validation images: {len(old)} (all three classes already predicted in step 3)")
    else:
        say(f"Validation images: {len(old)} Normal/Opacity (already predicted) + {len(nn)} Not-Normal (new)")

    model = tf.keras.models.load_model(str(args.run / "best.keras"), compile=False)
    lm = logit_model(model)

    # ------------------------------------------------------------------ 1. sanity check
    chk_ids = old.sample(64, random_state=SEED)["patientId"]
    chk = meta[meta["patientId"].isin(chk_ids)].reset_index(drop=True)
    z_new = predict_logits(lm, chk, args.split, backbone, img_size, args.batch_size)
    z_old = chk[["patientId"]].merge(old[["patientId", "logit"]], on="patientId")["logit"].values
    diff = float(np.max(np.abs(z_new - z_old)))
    say(f"Sanity check on 64 known images: max logit difference = {diff:.4f} "
        + ("(OK)" if diff < 0.05 else "(!! TOO LARGE - images are not prepared as in training; stop here)"))

    # ------------------------------------------------------------------ 2. predict Not-Normal
    keep = ["patientId", "class", "target", "view", "sex", "age", "logit"]
    if already:
        d = old[keep].copy()
    else:
        nn["logit"] = predict_logits(lm, nn, args.split, backbone, img_size, args.batch_size)
        d = pd.concat([old[keep], nn[keep]], ignore_index=True)
    views = d["view"].values.astype(object)
    d["prob"] = cu.probability(cal, d["logit"].values, views)   # calibrated, as in the app
    d["flag"] = cu.decide(cal, d["logit"].values, views)
    d = d.merge(meta[["patientId", "png_path"]], on="patientId", how="left")

    d["band"] = [cu.band(cal, p) for p in d["prob"]]
    d.drop(columns="png_path").to_csv(out_dir / "val_all_classes_predictions.csv", index=False)

    # ------------------------------------------------------------------ 3a. flagged rates
    say(); say("=" * 76); say(f"A. SHARE FLAGGED AS PNEUMONIA ({rule})"); say("=" * 76)
    say(f"{'class':12s}{'view':6s}{'n':>7s}{'flagged':>9s}{'rate':>8s}{'95 % CI':>18s}   meaning")
    meaning = {"normal": "false alarms", "not_normal": "false alarms", "opacity": "sensitivity"}
    for c in CLASSES:
        for v in ("all", "AP", "PA"):
            s = d[(d["class"] == c) & ((d["view"] == v) | (v == "all"))]
            k, n = int(s["flag"].sum()), len(s)
            lo, hi = wilson(k, n)
            say(f"{c:12s}{v:6s}{n:7d}{k:9d}{k / max(n, 1):8.3f}{f'{lo:.3f}-{hi:.3f}':>18s}   {meaning[c]}")
        say()

    # ------------------------------------------------------------------ 3b. bands
    say("=" * 76); say("B. CONFIDENCE BANDS — % of each class in each band"); say("=" * 76)
    tab = pd.crosstab(d["class"], d["band"], normalize="index").mul(100)
    tab = tab[[b["name"] for b in cal["bands"] if b["name"] in tab.columns]]
    say(tab.round(1).to_string())

    # ------------------------------------------------------------------ 3c. AUROC
    say(); say("=" * 76); say("C. AUROC — Opacity (positive) against each kind of negative"); say("=" * 76)
    say(f"{'negatives':24s}{'overall':>9s}{'AP':>8s}{'PA':>8s}{'within-view':>13s}")
    aucs = {}
    for name, negs in (("Normal", ("normal",)), ("Not-Normal", ("not_normal",)),
                       ("Normal + Not-Normal", ("normal", "not_normal"))):
        a = auc_pair(d, negs); aucs[name] = a
        say(f"{name:24s}{a['all']:9.3f}{a['AP']:8.3f}{a['PA']:8.3f}{a['within_view']:13.3f}")

    # ------------------------------------------------------------------ 3d. real-world mix
    say(); say("=" * 76); say("D. REAL-WORLD MIX (all three classes at their natural proportions)"); say("=" * 76)
    y, f = d["target"].values, d["flag"].values
    tp = int(((f == 1) & (y == 1)).sum()); fn = int(((f == 0) & (y == 1)).sum())
    fp = int(((f == 1) & (y == 0)).sum()); tn = int(((f == 0) & (y == 0)).sum())
    say(f"images {len(d)} | opacity {int(y.sum())} ({100 * y.mean():.1f} %)")
    say(f"sensitivity {tp / (tp + fn):.3f} | specificity {tn / (tn + fp):.3f} | "
        f"PPV {tp / max(tp + fp, 1):.3f} | NPV {tn / max(tn + fn, 1):.3f}")
    say(f"false alarms {fp}: {int(((f == 1) & (d['class'] == 'normal')).sum())} from Normal, "
        f"{int(((f == 1) & (d['class'] == 'not_normal')).sum())} from Not-Normal")
    say(f"-> of every 100 'pneumonia' answers, about {100 * fp / max(tp + fp, 1):.0f} would be wrong")

    # ------------------------------------------------------------------ 4. gallery
    top = d[(d["class"] == "not_normal")].sort_values("prob", ascending=False).head(16)
    try:
        gallery(top, out_dir / "top_false_alarms.png")
        say(f"\nGallery of the 16 most confident Not-Normal false alarms: {out_dir / 'top_false_alarms.png'}")
    except Exception as e:            # a missing image must not lose the numbers above
        say(f"\n(gallery skipped: {e})")

    summary = {"decision": cal["decision"], "calibration": cal["method"], "sanity_max_logit_diff": diff,
               "flag_rate": {c: float(d.loc[d["class"] == c, "flag"].mean()) for c in CLASSES},
               "auroc": aucs,
               "real_world": {"sens": tp / (tp + fn), "spec": tn / (tn + fp), "ppv": tp / max(tp + fp, 1)}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out_dir / "not_normal_report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved report, predictions and gallery in {out_dir}")


if __name__ == "__main__":
    main()
