"""
Step 4b — Per-view calibration and the decision policy (replaces step 4 for the final model)
============================================================================================

Input : a run folder (only val_predictions.csv is needed: logit, label and view per image)
Output (in the run folder):
    calibration_v2.json           Platt parameters per view, decision policy + threshold(s), bands
                                  -> read through calibration_utils.py by every later step
    val_predictions_v2.csv        val_predictions.csv + calibrated probability, decision, band
    calibration_v2_report.txt     the tables printed below
    reliability_v2.png            reliability diagrams for AP and PA, before and after

What it does, in order
  1. COMPARES four ways of turning a logit into a probability, honestly: each is fitted on
     4/5 of the validation images and scored on the remaining 1/5, five times over
     (5-fold cross-validation). So a method with more parameters cannot win just by
     memorising the validation set.
        none         sigmoid(logit)                        0 parameters
        temperature  sigmoid(logit / T)                    1   (what step 4 used)
        platt        sigmoid(a * logit + b)                2
        platt_view   sigmoid(a_v * logit + b_v), v = AP/PA 4
     Scores: log-loss (lower = better probabilities), ECE overall and within AP and PA,
     and the average predicted probability vs the real pneumonia rate in each view.
  2. FITS the chosen method (default platt_view) on all validation images.
  3. DECISION: compares three yes/no policies, each set to catch 90 % of pneumonia overall:
        neutral  : one threshold on the view-blind score: the same image evidence gets the
                   same decision whatever the view                             (default)
        per_view : one threshold per view, so 90 % is caught in AP AND in PA
        single   : one threshold on the view-honest probability (shown only for comparison)
     and saves the chosen one.
  4. BANDS: fixed probability cut-offs (0.10 / 0.30 / 0.70 / 0.90) that mean the same thing
     in every view, with the real pneumonia rate measured in each band.
  5. Bootstrap 95 % confidence intervals for sensitivity and specificity.

Run (inside WSL):
    python 04b_calibrate_per_view.py --run "<outputs>/runs/densenet121_384_fullft_20261027-1316"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from calibration_utils import VIEWS, band, decide, probability, sigmoid

SEED = 42
EPS = 1e-12
BAND_CUTS = [("very_unlikely", 0.0, 0.10), ("unlikely", 0.10, 0.30), ("uncertain", 0.30, 0.70),
             ("likely", 0.70, 0.90), ("very_likely", 0.90, 1.0000001)]


# ---------------------------------------------------------------------------- measures
def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, n_bins=10):
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean())
                     for k in range(n_bins) if (b == k).any()))


# ---------------------------------------------------------------------------- calibrators
def fit_platt(z, y):
    """Logistic regression on the logit: returns (a, b) of sigmoid(a*z + b). C=1e6 = no penalty."""
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def fit_temperature(z, y):
    best_t, best = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.2), np.log(5), 400)):        # simple, robust grid search
        l = log_loss(y, sigmoid(z / t))
        if l < best:
            best_t, best = t, l
    return float(best_t)


def fit(method, z, y, v):
    """Returns a calibration dict in the format calibration_utils expects (params only)."""
    if method == "none":
        params = {"overall": {"a": 1.0, "b": 0.0}}
    elif method == "temperature":
        t = fit_temperature(z, y)
        params = {"overall": {"a": 1.0 / t, "b": 0.0}}
    elif method == "platt":
        a, b = fit_platt(z, y)
        params = {"overall": {"a": a, "b": b}}
    elif method == "platt_view":
        a, b = fit_platt(z, y)
        params = {"overall": {"a": a, "b": b}}           # fallback when the view is unknown
        for k in VIEWS:
            ak, bk = fit_platt(z[v == k], y[v == k])
            params[k] = {"a": ak, "b": bk}
    else:
        raise ValueError(method)
    return {"method": method, "params": params}


# ---------------------------------------------------------------------------- thresholds
def thr_for_sens(y, p, target):
    pos = np.sort(p[y == 1])[::-1]
    return float(pos[int(np.ceil(target * len(pos))) - 1])


def rates(y, f):
    return dict(sens=float(f[y == 1].mean()), spec=float((~f[y == 0]).mean()),
                missed=int((~f[y == 1]).sum()), false_alarms=int(f[y == 0].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--method", default="platt_view", choices=["none", "temperature", "platt", "platt_view"])
    ap.add_argument("--policy", default="neutral", choices=["per_view", "neutral", "single"])
    ap.add_argument("--target-sens", type=float, default=0.90)
    args = ap.parse_args()

    df = pd.read_csv(args.run / "val_predictions.csv")
    z, y, v = df["logit"].values.astype(float), df["target"].values.astype(int), df["view"].values.astype(object)

    lines = []
    def say(s=""):
        print(s); lines.append(s)

    # ------------------------------------------------------------ 1. honest comparison (5-fold CV)
    say("=" * 100)
    say("1. CALIBRATION METHODS, 5-fold cross-validated on the validation set (lower is better)")
    say("=" * 100)
    say(f"{'method':12s}{'params':>7s}{'log-loss':>10s}{'ECE':>8s}{'ECE AP':>8s}{'ECE PA':>8s}"
        f"{'mean p AP':>11s}{'mean p PA':>11s}")
    strata = np.array([f"{a}_{b}" for a, b in zip(y, v)])
    cv = {}
    for method, k in (("none", 0), ("temperature", 1), ("platt", 2), ("platt_view", 4)):
        P = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(z, strata):
            P[te] = probability(fit(method, z[tr], y[tr], v[tr]), z[te], v[te])
        cv[method] = {"log_loss": log_loss(y, P), "ece": ece(y, P),
                      "ece_AP": ece(y[v == "AP"], P[v == "AP"]), "ece_PA": ece(y[v == "PA"], P[v == "PA"])}
        say(f"{method:12s}{k:7d}{cv[method]['log_loss']:10.4f}{cv[method]['ece']:8.4f}"
            f"{cv[method]['ece_AP']:8.4f}{cv[method]['ece_PA']:8.4f}"
            f"{P[v == 'AP'].mean():11.3f}{P[v == 'PA'].mean():11.3f}")
    say(f"{'(truth)':12s}{'':7s}{'':10s}{'':8s}{'':8s}{'':8s}{y[v == 'AP'].mean():11.3f}{y[v == 'PA'].mean():11.3f}")
    say(f"Chosen: {args.method}")

    # ------------------------------------------------------------ 2. fit on everything
    cal = fit(args.method, z, y, v)
    p = probability(cal, z, v)
    say(); say("Fitted parameters (probability = sigmoid(a * logit + b)):")
    for k, prm in cal["params"].items():
        say(f"  {k:8s} a = {prm['a']:.4f}   b = {prm['b']:+.4f}")

    # ------------------------------------------------------------ 3. decision policy
    say(); say("=" * 100)
    say(f"2. DECISION POLICIES, each set to catch {100 * args.target_sens:.0f} % of all pneumonia cases")
    say("=" * 100)
    o = cal["params"]["overall"]
    p_blind = sigmoid(o["a"] * z + o["b"])                 # view-blind score (overall Platt)
    th_single = thr_for_sens(y, p, args.target_sens)
    th_neutral = thr_for_sens(y, p_blind, args.target_sens)
    th_view = {k: thr_for_sens(y[v == k], p[v == k], args.target_sens) for k in VIEWS}
    f_single = p >= th_single
    f_neutral = p_blind >= th_neutral
    f_view = np.array([pi >= th_view.get(vi, th_single) for pi, vi in zip(p, v)])
    say(f"{'policy':12s}{'thresholds':>26s}{'AP sens':>9s}{'AP spec':>9s}{'PA sens':>9s}{'PA spec':>9s}"
        f"{'spec':>7s}{'missed':>8s}{'false+':>8s}")
    for name, f, th in (("per_view", f_view, f"AP {th_view['AP']:.3f} / PA {th_view['PA']:.3f}"),
                        ("neutral", f_neutral, f"{th_neutral:.3f} (view-blind)"),
                        ("single", f_single, f"{th_single:.3f}")):
        ra, rp, r = rates(y[v == "AP"], f[v == "AP"]), rates(y[v == "PA"], f[v == "PA"]), rates(y, f)
        say(f"{name:12s}{th:>26s}{ra['sens']:9.3f}{ra['spec']:9.3f}{rp['sens']:9.3f}{rp['spec']:9.3f}"
            f"{r['spec']:7.3f}{r['missed']:8d}{r['false_alarms']:8d}")
    say("per_view: same sensitivity in AP and PA | neutral: one cut on the view-blind score |")
    say("single: one cut on the view-honest probability (penalises PA - shown for comparison)")
    say(f"Chosen policy: {args.policy}")

    cal["decision"] = {"policy": args.policy, "target_sensitivity": args.target_sens,
                       "thresholds": ({**th_view, "overall": th_single} if args.policy == "per_view"
                                      else {"overall": th_neutral} if args.policy == "neutral"
                                      else {"overall": th_single})}
    cal["bands"] = [{"name": n, "low": a, "high": min(b, 1.0)} for n, a, b in BAND_CUTS]
    cal["bands"][-1]["high"] = 1.0000001
    dec = decide(cal, z, v).astype(bool)

    # ------------------------------------------------------------ 4. bands + uncertainty
    say(); say("=" * 100); say("3. CONFIDENCE BANDS (fixed cut-offs, the same in every view)"); say("=" * 100)
    df["prob_v2"], df["decision_v2"] = p, dec.astype(int)
    df["band_v2"] = [band(cal, pi) for pi in p]
    say(f"{'band':15s}{'range':>14s}{'images':>8s}{'really pneumonia':>18s}{'AP':>8s}{'PA':>8s}")
    def rate(mask):                       # real pneumonia rate among the images in `mask`
        return f"{100 * df.loc[mask, 'target'].mean():.0f}%" if mask.any() else "-"

    for b in cal["bands"]:
        m = df["band_v2"] == b["name"]
        rng_txt = f"{b['low']:.2f}-{min(b['high'], 1):.2f}"
        say(f"{b['name']:15s}{rng_txt:>14s}{int(m.sum()):8d}{rate(m):>18s}"
            f"{rate(m & (df['view'] == 'AP')):>8s}{rate(m & (df['view'] == 'PA')):>8s}")

    rng = np.random.default_rng(SEED)
    sens_b, spec_b = [], []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        sens_b.append(dec[i][y[i] == 1].mean()); spec_b.append((~dec[i][y[i] == 0]).mean())
    r = rates(y, dec)
    ci = {"sens": [float(np.percentile(sens_b, 2.5)), float(np.percentile(sens_b, 97.5))],
          "spec": [float(np.percentile(spec_b, 2.5)), float(np.percentile(spec_b, 97.5))]}
    say(); say(f"Chosen setup: sensitivity {r['sens']:.3f} (95 % CI {ci['sens'][0]:.3f}-{ci['sens'][1]:.3f}), "
               f"specificity {r['spec']:.3f} ({ci['spec'][0]:.3f}-{ci['spec'][1]:.3f}), "
               f"accuracy {float((dec == (y == 1)).mean()):.3f}")

    # ------------------------------------------------------------ save
    cal["cv_comparison"] = cv
    cal["val_metrics"] = {**r, "ci95": ci, "accuracy": float((dec == (y == 1)).mean())}
    cal["how_to_use"] = "calibration_utils.probability(cal, logit, view) then decide(cal, prob, view)"
    (args.run / "calibration_v2.json").write_text(json.dumps(cal, indent=2))
    df.to_csv(args.run / "val_predictions_v2.csv", index=False)
    (args.run / "calibration_v2_report.txt").write_text("\n".join(lines) + "\n")
    reliability(y, v, sigmoid(z), p, args.run / "reliability_v2.png")
    print(f"\nSaved calibration_v2.json, val_predictions_v2.csv, calibration_v2_report.txt, reliability_v2.png in {args.run}")


def reliability(y, v, p_raw, p_cal, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), dpi=130)
    for ax, k in zip(axes, VIEWS):
        m = v == k
        ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
        for p, name, col in ((p_raw, "before (raw sigmoid)", "#d1495b"), (p_cal, "after (per-view Platt)", "#2f5597")):
            pp, yy = p[m], y[m]
            b = np.minimum((pp * 10).astype(int), 9)
            xs = [pp[b == i].mean() for i in range(10) if (b == i).sum() >= 5]
            ys = [yy[b == i].mean() for i in range(10) if (b == i).sum() >= 5]
            ax.plot(xs, ys, marker="o", color=col, lw=1.6, label=f"{name}, ECE {ece(yy, pp):.3f}")
        ax.set_title(f"{k} images (n = {int(m.sum())}, pneumonia rate {y[m].mean():.2f})", fontsize=9)
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed pneumonia rate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    main()
