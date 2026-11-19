"""
Step 7 — Compare runs: one table, paired tests, seed spread, ensembles
======================================================================

Every experiment ends with the same question: is the new run really better than the
reference, or is the difference luck? This script answers it the same way every time.

Input : two or more run folders that have been through step 4
        (each needs val_predictions.csv from step 3; val_predictions_calibrated.csv from the
        old step 4 is used when it exists. Both are a single increasing function of the logit,
        so every AUROC / specificity number is identical either way; only ECE differs)
Output: printed tables + compare_<name>.txt next to the first run folder

What it does
  1. Summary row per run: within-view AUROC, overall AUROC, AUROC on AP and PA,
     specificity and false alarms at 90 % sensitivity, PA sensitivity, ECE.
  2. Paired bootstrap of every run against the REFERENCE run (default: the first one):
     the same validation images are resampled 2,000 times and both runs are scored on
     each resample. If the 95 % interval of the difference excludes 0, it is real.
  3. --spread : mean, standard deviation and range over all runs given
     (experiment A1: the same set-up trained with different seeds).
  4. --ensemble : averages the calibrated probabilities of all runs given and adds the
     ensemble as an extra row (experiment B2).

Run (inside WSL), from the runs folder:
    python 07_compare_runs.py --runs densenet121_384_20261025-1548 densenet121_384_seed7_... \\
                              densenet121_384_seed123_... --spread --ensemble --name seeds
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

N_BOOT = 2000
TARGET_SENS = 0.90
SEED = 0


def ece(y, p, n_bins=10):
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean())
                     for k in range(n_bins) if (b == k).any()))


def thr_for_sens(y, p, target=TARGET_SENS):
    pos = np.sort(p[y == 1])[::-1]
    return float(pos[int(np.ceil(target * len(pos))) - 1])


def metrics(y, p, v):
    """All summary numbers for one run (y labels, p calibrated probs, v views)."""
    out = {"auc": roc_auc_score(y, p)}
    for k in ("AP", "PA"):
        out[f"auc_{k}"] = roc_auc_score(y[v == k], p[v == k])
    out["view_auc"] = (out["auc_AP"] + out["auc_PA"]) / 2
    t = thr_for_sens(y, p)
    d = p >= t
    out["thr"] = t
    out["spec"] = float((~d[y == 0]).mean())
    out["false_alarms"] = int(d[y == 0].sum())
    out["sens_PA"] = float(d[(y == 1) & (v == "PA")].mean())
    out["ece"] = ece(y, p)
    return out


def boot_diff(y, v, p_ref, p_new, n=N_BOOT, seed=SEED):
    """Paired bootstrap of (new - ref) for within-view AUROC and specificity at 90 % sens.
    The threshold of each run is fixed (chosen on the full validation set), as in deployment."""
    rng = np.random.default_rng(seed)
    t_ref, t_new = thr_for_sens(y, p_ref), thr_for_sens(y, p_new)
    res = {"view_auc": [], "spec": [], "sens_PA": []}
    idx = np.arange(len(y))
    for _ in range(n):
        i = rng.choice(idx, len(idx))
        yi, vi = y[i], v[i]
        try:
            va = lambda p: np.mean([roc_auc_score(yi[vi == k], p[i][vi == k]) for k in ("AP", "PA")])
            res["view_auc"].append(va(p_new) - va(p_ref))
        except ValueError:
            pass
        neg = yi == 0
        res["spec"].append((p_new[i][neg] < t_new).mean() - (p_ref[i][neg] < t_ref).mean())
        pa = (yi == 1) & (vi == "PA")
        res["sens_PA"].append((p_new[i][pa] >= t_new).mean() - (p_ref[i][pa] >= t_ref).mean())
    return {k: (float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
            for k, x in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--ref", type=Path, default=None, help="reference run (default: the first)")
    ap.add_argument("--spread", action="store_true", help="mean / SD / range over all runs (seed experiment)")
    ap.add_argument("--ensemble", action="store_true", help="add the average of all runs as an extra row")
    ap.add_argument("--name", default="runs", help="name for the saved report")
    args = ap.parse_args()

    preds = {}
    for r in args.runs:
        f = r / "val_predictions_calibrated.csv"
        if f.exists():
            d = pd.read_csv(f)
        elif (r / "val_predictions.csv").exists():
            d = pd.read_csv(r / "val_predictions.csv")
            d["prob_cal"] = 1 / (1 + np.exp(-d["logit"].values))     # raw model probability
            print(f"note: {r.name} has no step-4 file; using its raw probabilities (ECE = uncalibrated)")
        else:
            raise SystemExit(f"{r / 'val_predictions.csv'} not found")
        preds[r.name] = d.set_index("patientId")
    names = list(preds)
    base = preds[names[0]]
    for n in names[1:]:
        if set(preds[n].index) != set(base.index):
            raise SystemExit(f"{n} was evaluated on different images - runs must share the same validation set")
    ids = base.index
    y = base.loc[ids, "target"].values.astype(int)
    v = base.loc[ids, "view"].values
    P = {n: preds[n].loc[ids, "prob_cal"].values for n in names}
    if args.ensemble and len(names) > 1:
        P["ENSEMBLE (mean of runs)"] = np.mean([P[n] for n in names], axis=0)
    ref = (args.ref.name if args.ref else names[0])

    lines = []
    def say(s=""):
        print(s); lines.append(s)

    say("=" * 108)
    say(f"1. SUMMARY  (validation, {len(y)} images; specificity / false alarms at the threshold giving 90 % sensitivity)")
    say("=" * 108)
    say(f"{'run':44s}{'view AUC':>9s}{'AUC':>7s}{'AP':>7s}{'PA':>7s}{'thr':>7s}{'spec':>7s}{'FA':>5s}{'PA sens':>8s}{'ECE':>7s}")
    M = {}
    for n, p in P.items():
        m = metrics(y, p, v); M[n] = m
        say(f"{n[:43]:44s}{m['view_auc']:9.4f}{m['auc']:7.4f}{m['auc_AP']:7.4f}{m['auc_PA']:7.4f}"
            f"{m['thr']:7.3f}{m['spec']:7.3f}{m['false_alarms']:5d}{m['sens_PA']:8.3f}{m['ece']:7.3f}")

    if args.spread and len(names) > 1:
        say()
        say("=" * 108); say(f"2. SPREAD over the {len(names)} runs (ensemble excluded)"); say("=" * 108)
        for k, label in (("view_auc", "within-view AUROC"), ("auc", "overall AUROC"), ("spec", "specificity @90% sens"),
                         ("sens_PA", "PA sensitivity"), ("ece", "ECE")):
            x = np.array([M[n][k] for n in names])
            say(f"{label:24s} mean {x.mean():.4f}   SD {x.std(ddof=1):.4f}   range {x.min():.4f} - {x.max():.4f}"
                f"   (max - min = {x.max() - x.min():.4f})")
        say("Rule of thumb: a difference between two set-ups smaller than about 2 x SD is not convincing,")
        say("even if a single paired test says it is.")

    say()
    say("=" * 108); say(f"3. PAIRED BOOTSTRAP vs reference: {ref}  ({N_BOOT} resamples; difference = run - reference)"); say("=" * 108)
    say(f"{'run':44s}{'within-view AUROC':>26s}{'specificity':>26s}{'PA sensitivity':>26s}")
    for n, p in P.items():
        if n == ref:
            continue
        d = boot_diff(y, v, P[ref], p)
        cell = lambda k: (f"{d[k][0]:+.4f} [{d[k][1]:+.3f},{d[k][2]:+.3f}]" + ("*" if d[k][1] > 0 or d[k][2] < 0 else " "))
        say(f"{n[:43]:44s}{cell('view_auc'):>26s}{cell('spec'):>26s}{cell('sens_PA'):>26s}")
    say("* = the 95 % interval excludes 0 (a real difference, not luck)")

    out = args.runs[0].parent / f"compare_{args.name}.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
