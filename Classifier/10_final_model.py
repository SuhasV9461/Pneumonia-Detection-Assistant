"""
Step 10 — The final model, and the ONE evaluation on the test set
=================================================================

Two stages, run one after the other. Nothing in stage 2 is allowed to change stage 1.

STAGE 1  --stage val   (validation only; can be re-run as often as you like)
  1. Predicts every validation image with each member model, once as it is and once for
     each of 4 slightly changed copies (test-time augmentation, TTA):
         original | rotated +5 deg | rotated -5 deg | zoomed in 6 % | zoomed out 6 %
     The model's logits for the 5 versions are averaged. A real finding should survive a
     small rotation or zoom; random noise in one prediction gets averaged away. The changes
     are half the size of the training augmentation (+-10 deg, +-10 % zoom), so every
     version looks like something the model has already seen. No left-right flip: the
     heart would end up on the wrong side.
     Sanity check: the "original" version must give the same logits as step 3 wrote in
     val_predictions.csv, otherwise the images are prepared differently from training.
  2. Compares, on validation:  each single model | the ensemble (average of the models)
                               | each model + TTA | the ensemble + TTA
     with the same table and paired bootstrap as step 7.
  3. Chooses the final configuration (the ensemble, with TTA only if it is not worse) and
     writes it to <outputs>/final_model/final_config.json. From now on it is FROZEN.
  4. Calibrates the final configuration exactly like step 4b (per-view Platt, neutral policy).

STAGE 2  --stage test  (runs ONCE; refuses to run again unless --force)
  5. Predicts the TEST images (Normal, Opacity AND Not-Normal) with the frozen configuration.
  6. Reports, with 95 % bootstrap confidence intervals:
       accuracy, sensitivity, specificity, PPV, NPV, F1, AUROC, within-view AUROC,
       per view (AP / PA), calibration (ECE), confidence bands
     a) on the whole test set (Normal + Opacity)
     b) on the test set WITHOUT the images that the leakage audit flagged as having a
        likely twin in the training set (shows whether leakage changes the result)
     c) the Not-Normal stress test: share of Not-Normal test X-rays flagged as pneumonia

Output: <outputs>/final_model/
    final_config.json, calibration_v2.json, val_predictions.csv, val_comparison.txt
    test_predictions.csv, TEST_REPORT.txt      (stage 2)
    cache_val.npz, cache_test.npz              (every logit, so nothing has to be predicted twice)

Run (inside WSL, from the code folder):
    python 10_final_model.py --stage val  --csv "<outputs>/metadata.csv" \\
        --runs "<runs>/densenet121_384_fullft_20261027-1316" "<runs>/densenet121_384_seed7_fullft_20261027-1342"
    python 10_final_model.py --stage test --csv "<outputs>/metadata.csv"
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
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from calibration_utils import band, decide, load, probability
from classifier_utils import logit_model, preprocess_fn
from data_pipeline import load_metadata, make_dataset

compare = importlib.import_module("07_compare_runs")    # metrics() and boot_diff() from step 7

SEED = 42
N_BOOT = 2000
# (name, rotation in degrees, zoom factor). Zoom > 1 = zoom in.
TTA = [("original", 0.0, 1.00), ("rot+5", 5.0, 1.00), ("rot-5", -5.0, 1.00),
       ("zoom_in", 0.0, 1.06), ("zoom_out", 0.0, 0.94)]


# ---------------------------------------------------------------------------- prediction
def affine(images, angle_deg, zoom):
    """Rotate / zoom a batch of images around its centre (black fill, as in training)."""
    if angle_deg == 0 and zoom == 1:
        return images
    h = tf.cast(tf.shape(images)[1], tf.float32)
    w = tf.cast(tf.shape(images)[2], tf.float32)
    a = angle_deg * math.pi / 180.0
    cos, sin = math.cos(a) / zoom, math.sin(a) / zoom      # output pixel -> input pixel
    cx, cy = w / 2.0, h / 2.0
    t = tf.stack([cos, -sin, cx - cos * cx + sin * cy, sin, cos, cy - sin * cx - cos * cy, 0.0, 0.0])
    t = tf.tile(t[None], [tf.shape(images)[0], 1])
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=images, transforms=t, output_shape=tf.shape(images)[1:3], fill_value=0.0,
        interpolation="BILINEAR", fill_mode="CONSTANT")


def predict_logits(run_dir, df, split, batch_size):
    """Logits of one member model for every TTA version: array (n_images, len(TTA))."""
    cfg = json.loads((run_dir / "config.json").read_text())
    model = tf.keras.models.load_model(str(run_dir / "best.keras"), compile=False)
    lm = logit_model(model)
    prep = preprocess_fn(cfg["backbone"])
    out = []
    for name, ang, zoom in TTA:
        ds = make_dataset(df, split, cfg["img_size"], batch_size, training=False, cache=False)
        ds = ds.map(lambda x, y: prep(affine(x, ang, zoom)), num_parallel_calls=tf.data.AUTOTUNE)
        print(f"  {run_dir.name} | {split} | {name}")
        out.append(lm.predict(ds, verbose=0).ravel().astype("float64"))
    del model, lm
    tf.keras.backend.clear_session()
    return np.stack(out, axis=1)


def combine(L, models, tta):
    """L: dict run -> (n, len(TTA)) logits. Average over the chosen models and TTA versions."""
    cols = list(range(len(TTA))) if tta else [0]
    return np.mean([L[m][:, cols].mean(axis=1) for m in models], axis=0)


# ---------------------------------------------------------------------------- statistics
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def star(d):
    """'*' when the 95 % interval (mean, low, high) of a difference excludes 0."""
    return " *" if d[1] > 0 or d[2] < 0 else "  "


def ece(y, p, n=10):
    b = np.minimum((p * n).astype(int), n - 1)
    return float(sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean()) for k in range(n) if (b == k).any()))


def summary(y, p, f, v):
    """All reported numbers for labels y, calibrated probabilities p, decisions f, views v."""
    tp = int((f & (y == 1)).sum()); fn = int((~f & (y == 1)).sum())
    fp = int((f & (y == 0)).sum()); tn = int((~f & (y == 0)).sum())
    m = {"n": len(y), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
         "accuracy": (tp + tn) / len(y), "sensitivity": tp / max(tp + fn, 1),
         "specificity": tn / max(tn + fp, 1), "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1),
         "f1": 2 * tp / max(2 * tp + fp + fn, 1), "auroc": roc_auc_score(y, p), "ece": ece(y, p)}
    for k in ("AP", "PA"):
        s = v == k
        m[f"auroc_{k}"] = roc_auc_score(y[s], p[s])
        m[f"sens_{k}"] = float(f[s & (y == 1)].mean())
        m[f"spec_{k}"] = float((~f[s & (y == 0)]).mean())
    m["within_view_auroc"] = (m["auroc_AP"] + m["auroc_PA"]) / 2
    return m


def bootstrap(y, p, f, v, keys, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    res = {k: [] for k in keys}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        try:
            s = summary(y[i], p[i], f[i], v[i])
        except ValueError:
            continue
        for k in keys:
            res[k].append(s[k])
    return {k: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) for k, x in res.items()}


# ---------------------------------------------------------------------------- stage 1
def stage_val(args, out):
    df = load_metadata(args.csv, include_not_normal=False)
    val = df[df["split"] == "val"].reset_index(drop=True)
    names = [r.name for r in args.runs]
    cache = out / "cache_val.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        L = {n: z[n] for n in names if n in z.files}
    else:
        L = {}
    for r in args.runs:
        if r.name not in L:
            L[r.name] = predict_logits(r, df, "val", args.batch_size)
    np.savez_compressed(cache, ids=val["patientId"].values, **L)

    # sanity check: untouched images must reproduce step 3's own validation logits
    for r in args.runs:
        old = pd.read_csv(r / "val_predictions.csv")[["patientId", "logit"]]
        chk = val[["patientId"]].assign(new=L[r.name][:, 0]).merge(old, on="patientId")
        diff = float(np.abs(chk["new"] - chk["logit"]).max())
        print(f"Sanity check {r.name}: {len(chk)} images, max logit difference vs step 3 = {diff:.4f} "
              + ("(OK)" if diff < 0.05 else "(!! TOO LARGE - stop and send me this)"))

    y, v = val["target"].values.astype(int), val["view"].values
    configs = {}
    for n in names:
        configs[f"{n[:30]}"] = combine(L, [n], False)
        configs[f"{n[:30]} + TTA"] = combine(L, [n], True)
    configs["ENSEMBLE"] = combine(L, names, False)
    configs["ENSEMBLE + TTA"] = combine(L, names, True)

    lines = []
    def say(s=""):
        print(s); lines.append(s)
    say("=" * 104)
    say(f"VALIDATION ({len(y)} images): single models, ensemble, and test-time augmentation ({len(TTA)} versions)")
    say("=" * 104)
    say(f"{'configuration':40s}{'view AUC':>9s}{'AUC':>7s}{'AP':>7s}{'PA':>7s}{'spec@90':>9s}{'FA':>5s}{'PA sens':>9s}")
    M = {}
    for k, z in configs.items():
        m = compare.metrics(y, sigmoid(z), v); M[k] = m
        say(f"{k:40s}{m['view_auc']:9.4f}{m['auc']:7.4f}{m['auc_AP']:7.4f}{m['auc_PA']:7.4f}"
            f"{m['spec']:9.3f}{m['false_alarms']:5d}{m['sens_PA']:9.3f}")
    say()
    say("Paired bootstrap (difference, 95 % interval; * = interval excludes 0):")
    ref = f"{names[0][:30]}"
    for k in ("ENSEMBLE", "ENSEMBLE + TTA"):
        d = compare.boot_diff(y, v, sigmoid(configs[ref]), sigmoid(configs[k]))
        say(f"  {k:16s} vs {ref}: within-view AUROC {d['view_auc'][0]:+.4f} "
            f"[{d['view_auc'][1]:+.4f}, {d['view_auc'][2]:+.4f}]{star(d['view_auc'])}  specificity "
            f"{d['spec'][0]:+.4f} [{d['spec'][1]:+.4f}, {d['spec'][2]:+.4f}]{star(d['spec'])}")
    d = compare.boot_diff(y, v, sigmoid(configs["ENSEMBLE"]), sigmoid(configs["ENSEMBLE + TTA"]))
    say(f"  TTA on top of the ensemble: within-view AUROC {d['view_auc'][0]:+.4f} "
        f"[{d['view_auc'][1]:+.4f}, {d['view_auc'][2]:+.4f}]{star(d['view_auc'])}  specificity "
        f"{d['spec'][0]:+.4f} [{d['spec'][1]:+.4f}, {d['spec'][2]:+.4f}]{star(d['spec'])}")

    # decision rule: the ensemble always (averaging never hurts on average); TTA only if it does not hurt
    use_tta = M["ENSEMBLE + TTA"]["view_auc"] >= M["ENSEMBLE"]["view_auc"]
    chosen = "ENSEMBLE + TTA" if use_tta else "ENSEMBLE"
    say(); say(f"CHOSEN: {chosen}   (rule: ensemble of all models; add TTA only if it is not worse on validation)")

    final_cfg = {"members": [str(r) for r in args.runs], "tta": [list(t) for t in (TTA if use_tta else TTA[:1])],
                 "combine": "mean of logits over members and TTA versions", "frozen_at": datetime.now().isoformat(timespec="seconds"),
                 "csv": str(args.csv)}
    (out / "final_config.json").write_text(json.dumps(final_cfg, indent=2))
    z = configs[chosen]
    vp = val[["patientId", "class", "target", "view", "sex", "age"]].copy()
    vp["logit"], vp["prob"] = z, sigmoid(z)
    vp.to_csv(out / "val_predictions.csv", index=False)
    (out / "val_comparison.txt").write_text("\n".join(lines) + "\n")

    # calibrate exactly like step 4b (per-view Platt + neutral policy by default)
    say(); say("Calibrating the final configuration with step 4b ...")
    subprocess.run([sys.executable, str(Path(__file__).parent / "04b_calibrate_per_view.py"),
                    "--run", str(out), "--policy", args.policy], check=True)
    print(f"\nStage 1 done. The final model is frozen in {out}. Next: --stage test (once).")


# ---------------------------------------------------------------------------- stage 2
def stage_test(args, out):
    report = out / "TEST_REPORT.txt"
    if report.exists() and not args.force:
        raise SystemExit(f"{report} already exists: the test set has been used. "
                         "Re-running would turn it into another validation set. (Use --force only if "
                         "the previous run crashed before finishing.)")
    final_cfg = json.loads((out / "final_config.json").read_text())
    cal = load(out / "calibration_v2.json")
    runs = [Path(r) for r in final_cfg["members"]]
    use_tta = len(final_cfg["tta"]) > 1

    df = load_metadata(args.csv, include_not_normal=True)          # opacity = 1, normal + not_normal = 0
    test = df[df["split"] == "test"].reset_index(drop=True)
    cache = out / "cache_test.npz"
    L = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
    for r in runs:
        if r.name not in L:
            L[r.name] = predict_logits(r, df, "test", args.batch_size)
    np.savez_compressed(cache, ids=test["patientId"].values, **{k: L[k] for k in [r.name for r in runs]})

    z = combine(L, [r.name for r in runs], use_tta)
    v = test["view"].values.astype(object)
    test["logit"] = z
    test["prob"] = probability(cal, z, v)
    test["decision"] = decide(cal, z, v)
    test["band"] = [band(cal, p) for p in test["prob"]]

    # likely twins from the leakage audit (flagged by either fingerprint at the 99 % level)
    twins = set()
    la = args.leakage or args.csv.parent / "leakage_audit"
    pairs_f, thr_f = la / "pairs.csv", la / "thresholds.json"
    if pairs_f.exists() and thr_f.exists():
        P, T = pd.read_csv(pairs_f), json.loads(thr_f.read_text())
        P = P[P["split"] == "test"]
        flag = (P["deep_cand_sim"] > T["deep"]["0.99"]) | (P["thumb_cand_sim"] > T["thumb"]["0.99"])
        twins = set(P.loc[flag, "patientId"])
    test["likely_twin_in_train"] = test["patientId"].isin(twins)
    test.drop(columns=["png_path"], errors="ignore").to_csv(out / "test_predictions.csv", index=False)

    lines = []
    def say(s=""):
        print(s); lines.append(s)
    keys = ["accuracy", "sensitivity", "specificity", "ppv", "npv", "f1", "auroc", "within_view_auroc"]

    def block(title, d):
        y = d["target"].values.astype(int); p = d["prob"].values
        f = d["decision"].values.astype(bool); vv = d["view"].values
        m = summary(y, p, f, vv); ci = bootstrap(y, p, f, vv, keys)
        say("=" * 90); say(title); say("=" * 90)
        say(f"images {m['n']} (pneumonia {int(y.sum())}, normal {int((y == 0).sum())})")
        say(f"confusion matrix: TP {m['tp']}  FN {m['fn']}  FP {m['fp']}  TN {m['tn']}")
        for k in keys:
            say(f"  {k:20s} {m[k]:.3f}   (95 % CI {ci[k][0]:.3f} - {ci[k][1]:.3f})")
        say(f"  {'AP':20s} sensitivity {m['sens_AP']:.3f}  specificity {m['spec_AP']:.3f}  AUROC {m['auroc_AP']:.3f}")
        say(f"  {'PA':20s} sensitivity {m['sens_PA']:.3f}  specificity {m['spec_PA']:.3f}  AUROC {m['auroc_PA']:.3f}")
        say(f"  {'calibration ECE':20s} {m['ece']:.3f}")
        return m, ci

    main = test[test["class"].isin(["normal", "opacity"])]
    say(f"FINAL MODEL: {len(runs)} member model(s), TTA {'on' if use_tta else 'off'}, "
        f"policy {cal['decision']['policy']}, threshold(s) {cal['decision']['thresholds']}")
    say(f"Evaluated {datetime.now().isoformat(timespec='seconds')}")
    m_all, ci_all = block("A. TEST SET (Normal vs Pneumonia)", main)
    say(); say("Confidence bands on the test set:")
    for b in cal["bands"]:
        s = main[main["band"] == b["name"]]
        rate = f"{100 * s['target'].mean():.0f} %" if len(s) else "-"
        say(f"  {b['name']:15s} {len(s):5d} images   really pneumonia {rate}")
    say()
    m_nt, ci_nt = (None, None)
    if twins:
        m_nt, ci_nt = block(f"B. TEST SET WITHOUT LIKELY TWINS ({int(main['likely_twin_in_train'].sum())} removed)",
                            main[~main["likely_twin_in_train"]])
        say(f"  change vs A: accuracy {m_nt['accuracy'] - m_all['accuracy']:+.3f}, "
            f"AUROC {m_nt['auroc'] - m_all['auroc']:+.4f}")
    else:
        say(f"B. skipped: {pairs_f} not found (point --leakage at the leakage_audit folder)")
    say()
    nn = test[test["class"] == "not_normal"]
    if len(nn):
        k, n = int(nn["decision"].sum()), len(nn)
        say("=" * 90); say("C. STRESS TEST: Not-Normal X-rays (abnormal, but not pneumonia; never seen in training)")
        say("=" * 90)
        say(f"  flagged as pneumonia: {k} of {n} ({100 * k / n:.1f} %)")
        y3 = np.r_[np.ones(int((main['target'] == 1).sum())), np.zeros(n)]
        p3 = np.r_[main.loc[main["target"] == 1, "prob"].values, nn["prob"].values]
        say(f"  AUROC pneumonia vs Not-Normal: {roc_auc_score(y3, p3):.3f}  (vs Normal: {m_all['auroc']:.3f})")

    results = {"test_all": m_all, "test_all_ci95": ci_all, "test_no_twins": m_nt, "test_no_twins_ci95": ci_nt,
               "final_config": final_cfg, "calibration": cal["decision"]}
    (out / "test_results.json").write_text(json.dumps(results, indent=2, default=float))
    report.write_text("\n".join(lines) + "\n")
    print(f"\nSaved TEST_REPORT.txt, test_results.json and test_predictions.csv in {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["val", "test"])
    ap.add_argument("--csv", type=Path, required=True, help="metadata.csv")
    ap.add_argument("--runs", nargs="+", type=Path, help="member run folders (stage val)")
    ap.add_argument("--out", type=Path, default=None, help="default: <csv folder>/final_model")
    ap.add_argument("--policy", default="neutral", choices=["neutral", "per_view", "single"])
    ap.add_argument("--leakage", type=Path, default=None, help="leakage_audit folder (default: next to metadata.csv)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="stage test: allow re-running after a crash")
    args = ap.parse_args()
    out = args.out or args.csv.parent / "final_model"
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "val":
        if not args.runs:
            raise SystemExit("--stage val needs --runs")
        stage_val(args, out)
    else:
        stage_test(args, out)


if __name__ == "__main__":
    main()
