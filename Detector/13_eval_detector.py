"""
Step 13 — Evaluate the detector the way the report will use it
==============================================================

Run in yolo-env (needs: pip install scikit-learn). Needs a detector run from step 12,
yolo_data/ from step 11 and the classifier's predictions from step 10.

mAP (step 12) only says how well boxes overlap the radiologists' boxes. The report needs
other answers, so this script measures them.

STAGE 1  --split val   (as often as you like)
  1. Predicts every validation image ONCE at a very low cut-off (confidence 0.01) and
     caches all boxes in predictions_val.csv; everything after that is cheap.
  2. Chooses the post-processing on validation, using the RSNA competition metric
     (per image: matches at overlap 0.40, 0.45 ... 0.75, score = TP / (TP + FP + FN);
     an image with no real box and no predicted box is skipped, a predicted box on a
     clean image scores 0 - so false boxes on Normal / Not-Normal X-rays are punished):
       - confidence cut-off: 0.05 ... 0.50
       - merging: a second box is dropped if it overlaps a stronger box by more than
         0.7 (YOLO's own default), 0.5 or 0.3. Fixes the duplicate boxes seen in step 12.
     The winner is FROZEN in <run>/detector_config.json.
  3. With the frozen settings, per image and per box:
       a. how many pneumonia X-rays get >= 1 box, and >= 1 CORRECT box (overlap >= 0.4)
       b. how many Normal and Not-Normal X-rays get a false box
       c. all of it by view, sex and age group
       d. localisation, per radiologist box: found at all? right lung found? right
          lung AND zone found?  (zones = thirds of the image height, as in the report
          figure; side = which half of the image, patient's right = image left)
       e. recall by box size (small / medium / large thirds of all box areas)
       f. your breast-tissue question: false boxes in the LOWER zone, female vs male
  4. PA rescue experiment (Normal vs Pneumonia only): the detector's best box score is
     added to the classifier's logit with a small logistic regression (5-fold cross-
     validation, repeated 5 times). Compared with the classifier alone at the same 90 %
     overall sensitivity: within-view AUROC, PA sensitivity, false alarms, paired
     bootstrap. Only if the gain is clear would we change the decision rule.
  5. Galleries: correct boxes, missed pneumonia, false boxes on Normal / Not-Normal
     (green = radiologist, red = detector, with its confidence).

STAGE 2  --split test  (ONCE, after everything is frozen; refuses to run twice)
  Same measurements (3 + 5) with the frozen settings on the test split.

Output: <run>/eval_<split>/   report.txt, predictions_<split>.csv (all raw boxes),
        image_level.csv, box_level.csv, gallery_*.png;  <run>/detector_config.json

Run (inside WSL, yolo-env active, from the code folder):
    pip install scikit-learn                     # once
    python 13_eval_detector.py --run "$HOME/AIML Project/Test 2/RSNA/outputs/detector_runs/yolo11s_512_20261028-0921"
"""

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

compare = importlib.import_module("07_compare_runs")    # metrics() and boot_diff() from step 7

RAW_CONF = 0.01
CONF_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
MERGE_GRID = [0.7, 0.5, 0.3]
RSNA_IOU = np.arange(0.40, 0.751, 0.05)
HIT_IOU = 0.40
SEED = 42


# ---------------------------------------------------------------------------- geometry
def iou_matrix(a, b):
    """a: (n,4), b: (m,4) boxes as x1 y1 x2 y2 (fractions of the image). Returns (n,m)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


def merge(boxes, conf, thr):
    """Keep the strongest box, drop every other box overlapping it by more than thr, repeat."""
    order = np.argsort(-conf)
    keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1:
            break
        ov = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ov <= thr]
    return np.array(keep, dtype=int)


def side(b):      # patient's right lung is on the image LEFT
    return np.where((b[:, 0] + b[:, 2]) / 2 < 0.5, "right", "left")


def zone(b):
    yc = (b[:, 1] + b[:, 3]) / 2
    return np.where(yc < 1 / 3, "upper", np.where(yc < 2 / 3, "middle", "lower"))


def rsna_score(gt, pr, conf):
    """RSNA competition score of one image, or None if it has no real and no predicted box."""
    if len(gt) == 0 and len(pr) == 0:
        return None
    if len(gt) == 0 or len(pr) == 0:
        return 0.0
    pr = pr[np.argsort(-conf)]
    ov = iou_matrix(pr, gt)
    s = []
    for t in RSNA_IOU:
        used = np.zeros(len(gt), bool); tp = 0
        for i in range(len(pr)):
            cand = np.where(~used & (ov[i] > t))[0]
            if len(cand):
                used[cand[np.argmax(ov[i, cand])]] = True; tp += 1
        fp = len(pr) - tp; fn = len(gt) - tp
        s.append(tp / (tp + fp + fn))
    return float(np.mean(s))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------------------- data
def read_labels(lab_dir, pid):
    f = lab_dir / f"{pid}.txt"
    rows = [list(map(float, l.split()))[1:] for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
    if not rows:
        return np.zeros((0, 4))
    a = np.array(rows)
    return np.c_[a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2]


def predict_all(weights, man, imgsz, out_csv):
    if out_csv.exists():
        print(f"Using cached predictions {out_csv.name}")
        return pd.read_csv(out_csv)
    from ultralytics import YOLO
    model = YOLO(str(weights))
    rec, paths = [], man["image"].tolist()
    for s in range(0, len(paths), 64):
        chunk = paths[s:s + 64]
        res = model.predict(chunk, imgsz=imgsz, conf=RAW_CONF, iou=0.7, max_det=20, augment=False,
                            verbose=False)
        for pid, r in zip(man["patientId"].iloc[s:s + 64], res):
            b = r.boxes.xyxyn.cpu().numpy(); c = r.boxes.conf.cpu().numpy()
            for bb, cc in zip(b, c):
                rec.append([pid, *bb.tolist(), float(cc)])
        print(f"\r  predicted {min(s + 64, len(paths))}/{len(paths)}", end="")
    print()
    P = pd.DataFrame(rec, columns=["patientId", "x1", "y1", "x2", "y2", "conf"])
    P.to_csv(out_csv, index=False)
    return P


def boxes_after(P_by, pid, conf_thr, merge_thr):
    d = P_by.get(pid)
    if d is None:
        return np.zeros((0, 4)), np.zeros(0)
    d = d[d["conf"] >= conf_thr]
    b, c = d[["x1", "y1", "x2", "y2"]].values, d["conf"].values
    k = merge(b, c, merge_thr) if len(b) else np.array([], int)
    return b[k], c[k]


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="detector run folder from step 12")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--clf", type=Path, default=None,
                    help="classifier predictions (default: <outputs>/final_model/<split>_predictions.csv)")
    ap.add_argument("--force", action="store_true", help="test: allow re-running after a crash")
    args = ap.parse_args()

    run = args.run
    rargs = yaml.safe_load((run / "args.yaml").read_text())
    data_dir = Path(rargs["data"]).parent
    outputs = data_dir.parent
    imgsz = int(rargs.get("imgsz", 512))
    out = run / f"eval_{args.split}"
    out.mkdir(exist_ok=True)
    cfg_f = run / "detector_config.json"
    if args.split == "test":
        if not cfg_f.exists():
            raise SystemExit("Run --split val first: the settings must be frozen on validation.")
        if (out / "report.txt").exists() and not args.force:
            raise SystemExit(f"{out / 'report.txt'} exists: the detector's test set has been used once already.")

    man = pd.read_csv(data_dir / "manifest.csv")
    man = man[man["split"] == args.split].reset_index(drop=True)
    # build paths from the dataset folder itself (still correct if the folder was moved)
    man["image"] = [str(data_dir / "images" / args.split / f"{p}.png") for p in man["patientId"]]
    lab_dir = data_dir / "labels" / args.split
    GT = {p: read_labels(lab_dir, p) for p in man["patientId"]}
    P = predict_all(run / "weights" / "best.pt", man, imgsz, out / f"predictions_{args.split}.csv")
    P_by = dict(tuple(P.groupby("patientId")))

    lines = []
    def say(s=""):
        print(s); lines.append(s)
    say(f"DETECTOR EVALUATION - {args.split} split - {run.name} - {datetime.now().isoformat(timespec='seconds')}")
    say(f"images {len(man)}: " + ", ".join(f"{k} {v}" for k, v in man["class"].value_counts().items()))

    # ------------------------------------------------------------ 1. post-processing
    if args.split == "val":
        say(); say("=" * 92)
        say("1. POST-PROCESSING chosen on validation by the RSNA metric (higher is better)")
        say("=" * 92)
        say(f"{'merge >':>8s}" + "".join(f"{c:>8.2f}" for c in CONF_GRID) + "   <- confidence cut-off")
        grid = {}
        for m in MERGE_GRID:
            row = []
            for c in CONF_GRID:
                sc = [rsna_score(GT[p], *boxes_after(P_by, p, c, m)) for p in man["patientId"]]
                sc = [x for x in sc if x is not None]
                grid[(m, c)] = float(np.mean(sc)); row.append(grid[(m, c)])
            say(f"{m:8.1f}" + "".join(f"{x:8.3f}" for x in row))
        m_best, c_best = max(grid, key=grid.get)
        cfg = {"conf": c_best, "merge_iou": m_best, "rsna_metric_val": grid[(m_best, c_best)],
               "imgsz": imgsz, "weights": str(run / "weights" / "best.pt"),
               "frozen_at": datetime.now().isoformat(timespec="seconds")}
        cfg_f.write_text(json.dumps(cfg, indent=2))
        say(f"CHOSEN: confidence >= {c_best:.2f}, drop boxes overlapping a stronger one by > {m_best:.1f}"
            f"   (RSNA metric {grid[(m_best, c_best)]:.3f}) -> frozen in detector_config.json")
    cfg = json.loads(cfg_f.read_text())
    C, M = cfg["conf"], cfg["merge_iou"]

    # ------------------------------------------------------------ 2. per image / per box
    img_rows, box_rows = [], []
    for r in man.itertuples(index=False):
        gt = GT[r.patientId]
        pb, pc = boxes_after(P_by, r.patientId, C, M)
        ov = iou_matrix(pb, gt)
        hit_pred = ov.max(axis=1) >= HIT_IOU if len(gt) and len(pb) else np.zeros(len(pb), bool)
        img_rows.append({"patientId": r.patientId, "class": r[man.columns.get_loc("class")], "view": r.view,
                         "sex": r.sex, "age": r.age, "n_gt": len(gt), "n_pred": len(pb),
                         "n_correct": int(hit_pred.sum()), "n_false": int((~hit_pred).sum()),
                         "max_conf": float(pc.max()) if len(pc) else 0.0,
                         "raw_max_conf": float(P_by[r.patientId]["conf"].max()) if r.patientId in P_by else 0.0,
                         "false_lower": int(((~hit_pred) & (zone(pb) == "lower")).sum()) if len(pb) else 0,
                         "rsna": rsna_score(gt, pb, pc)})
        if len(gt):
            ps, pz = side(pb), zone(pb)
            for j, (g, gs, gz) in enumerate(zip(gt, side(gt), zone(gt))):
                box_rows.append({"patientId": r.patientId, "view": r.view, "sex": r.sex,
                                 "area_pct": 100 * (g[2] - g[0]) * (g[3] - g[1]), "side": gs, "zone": gz,
                                 "found": bool(len(pb) and ov[:, j].max() >= HIT_IOU),
                                 "side_found": bool((ps == gs).any()),
                                 "side_zone_found": bool(((ps == gs) & (pz == gz)).any())})
    I = pd.DataFrame(img_rows); B = pd.DataFrame(box_rows)
    I["agegrp"] = pd.cut(I["age"], [0, 18, 40, 60, 200], right=False, labels=["<18", "18-39", "40-59", "60+"])
    I.to_csv(out / "image_level.csv", index=False); B.to_csv(out / "box_level.csv", index=False)

    say(); say("=" * 92)
    say(f"2. PER IMAGE at the frozen settings (confidence >= {C:.2f}, merge > {M:.1f})")
    say("=" * 92)
    sc = I["rsna"].dropna()
    say(f"RSNA metric on this split: {sc.mean():.3f}  ({len(sc)} images scored)")
    say(f"{'group':26s}{'n':>6s}{'>=1 box':>10s}{'>=1 correct':>13s}{'false boxes / image':>21s}")

    def rate(d, title):
        n = len(d); k1 = int((d.n_pred > 0).sum()); k2 = int((d.n_correct > 0).sum())
        lo, hi = wilson(k1, n)
        corr = f"{k2 / n:13.3f}" if d["n_gt"].sum() else f"{'-':>13s}"
        say(f"{title:26s}{n:6d}{k1 / max(n, 1):10.3f}{corr}{d.n_false.mean():21.3f}"
            f"   [95 % CI of '>=1 box' {lo:.3f}-{hi:.3f}]")
    for c in ("opacity", "normal", "not_normal"):
        d = I[I["class"] == c]
        if not len(d):
            continue
        rate(d, c)
        for v in ("AP", "PA"):
            rate(d[d.view == v], f"   {v}")
        for s in ("F", "M"):
            rate(d[d.sex == s], f"   sex {s}")
        for g in ["<18", "18-39", "40-59", "60+"]:
            dd = d[d.agegrp == g]
            if len(dd):
                rate(dd, f"   age {g}")
        say()
    say("For opacity: '>=1 box' = the detector marks something; '>=1 correct' = at least one box")
    say(f"overlaps a radiologist's box by >= {HIT_IOU}. For normal / not_normal every box is false.")

    # ------------------------------------------------------------ 3. localisation
    say(); say("=" * 92)
    say("3. LOCALISATION, per radiologist box (what the report will say: which lung, which zone)")
    say("=" * 92)
    if len(B):
        def loc(d, title):
            say(f"{title:26s}{len(d):6d}{d.found.mean():10.3f}{d.side_found.mean():12.3f}{d.side_zone_found.mean():14.3f}")
        say(f"{'group':26s}{'boxes':>6s}{'found':>10s}{'lung found':>12s}{'lung + zone':>14s}")
        loc(B, "all radiologist boxes")
        for v in ("AP", "PA"):
            loc(B[B.view == v], f"   {v}")
        B["size"] = pd.qcut(B["area_pct"], 3, labels=["small", "medium", "large"])
        for s in ("small", "medium", "large"):
            d = B[B["size"] == s]
            loc(d, f"   {s} ({d.area_pct.min():.1f}-{d.area_pct.max():.1f} % of image)")
        for z in ("upper", "middle", "lower"):
            loc(B[B.zone == z], f"   {z} zone")
        bil = I[I.n_gt >= 2]
        k = (B[B.patientId.isin(bil.patientId)].groupby("patientId")["found"].sum() >= 2).mean()
        say(f"Pneumonia in BOTH lungs ({len(bil)} images): both sides found in {100 * k:.0f} %")
        say("found = a detector box overlaps it by >= 0.4 | lung found = any box on that side |")
        say("lung + zone = any box on that side AND in that third of the image")

    # ------------------------------------------------------------ 4. breast-tissue question
    say(); say("=" * 92)
    say("4. FALSE BOXES IN THE LOWER ZONE, female vs male (Normal + Not-Normal X-rays)")
    say("=" * 92)
    neg = I[I["class"] != "opacity"]
    for v in ("AP", "PA", "all"):
        say(f"{v:4s} " + "   ".join(
            f"{s}: {(lambda d: f'{(d.false_lower > 0).mean():.3f} of {len(d)} images')(neg[(neg.sex == s) & ((neg.view == v) | (v == 'all'))])}"
            for s in ("F", "M")))
    say("If female lower-zone false boxes were clearly higher, breast shadow would be the suspect.")

    # ------------------------------------------------------------ 5. PA rescue experiment
    clf_f = args.clf or outputs / "final_model" / f"{args.split}_predictions.csv"
    if clf_f.exists():
        say(); say("=" * 92)
        say("5. PA RESCUE: does the detector's best box score help the classifier? (Normal vs Pneumonia)")
        say("=" * 92)
        clf = pd.read_csv(clf_f)[["patientId", "logit"]]
        D = I[I["class"].isin(["normal", "opacity"])].merge(clf, on="patientId")
        y, v = (D["class"] == "opacity").astype(int).values, D["view"].values
        det = np.log(np.clip(D["raw_max_conf"].values, 1e-3, 1 - 1e-3) / (1 - np.clip(D["raw_max_conf"].values, 1e-3, 1 - 1e-3)))
        X = np.c_[D["logit"].values, det, (v == "AP").astype(float)]
        X = np.c_[X, X[:, 1] * X[:, 2]]
        comb = np.zeros(len(y))
        for rep in range(5):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
                comb[te] += LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr]).decision_function(X[te]) / 5
        sig = lambda z: 1 / (1 + np.exp(-z))
        say(f"images {len(D)} | classifier = the frozen Phase 1 ensemble logit | detector = best raw box score")
        say(f"{'':28s}{'view AUC':>9s}{'AP':>7s}{'PA':>7s}{'spec@90':>9s}{'FA':>5s}{'PA sens':>9s}")
        for name, z in (("classifier alone", D["logit"].values), ("detector alone", det),
                        ("classifier + detector (CV)", comb)):
            m = compare.metrics(y, sig(z), v)
            say(f"{name:28s}{m['view_auc']:9.4f}{m['auc_AP']:7.4f}{m['auc_PA']:7.4f}{m['spec']:9.3f}"
                f"{m['false_alarms']:5d}{m['sens_PA']:9.3f}")
        d = compare.boot_diff(y, v, sig(D["logit"].values), sig(comb))
        say(f"combined - classifier: within-view AUROC {d['view_auc'][0]:+.4f} [{d['view_auc'][1]:+.4f}, "
            f"{d['view_auc'][2]:+.4f}] | specificity {d['spec'][0]:+.4f} [{d['spec'][1]:+.4f}, {d['spec'][2]:+.4f}] "
            f"| PA sensitivity {d['sens_PA'][0]:+.3f} [{d['sens_PA'][1]:+.3f}, {d['sens_PA'][2]:+.3f}]")
        say("Adopt only if the intervals exclude 0 in our favour; otherwise the classifier stays the decision-maker.")
    else:
        say(f"\n5. skipped: {clf_f} not found")

    # ------------------------------------------------------------ 6. galleries
    def gallery(rows, title, path):
        tiles = []
        for r in rows.itertuples(index=False):
            img = cv2.imread(man.loc[man.patientId == r.patientId, "image"].iloc[0])
            if img is None:
                continue
            img = cv2.resize(img, (320, 320))
            for g in GT[r.patientId]:
                cv2.rectangle(img, tuple((g[:2] * 320).astype(int)), tuple((g[2:] * 320).astype(int)), (0, 200, 0), 2)
            pb, pc = boxes_after(P_by, r.patientId, C, M)
            for b, c in zip(pb, pc):
                cv2.rectangle(img, tuple((b[:2] * 320).astype(int)), tuple((b[2:] * 320).astype(int)), (0, 0, 255), 2)
                cv2.putText(img, f"{c:.2f}", (int(b[0] * 320) + 2, int(b[1] * 320) + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.putText(img, f"{r.view} {r.sex} {r.age:.0f}", (4, 314), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(img)
        if not tiles:
            return
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        bar = np.full((30, grid.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imwrite(str(path), np.vstack([bar, grid]))

    rs = dict(random_state=SEED)
    op = I[I["class"] == "opacity"]
    gallery(op[op.n_correct > 0].sample(min(8, int((op.n_correct > 0).sum())), **rs),
            "correct: green = radiologist, red = detector", out / "gallery_correct.png")
    gallery(op[op.n_pred == 0].sample(min(8, int((op.n_pred == 0).sum())), **rs),
            "missed pneumonia: no detector box", out / "gallery_missed.png")
    for c in ("normal", "not_normal"):
        d = I[(I["class"] == c) & (I.n_pred > 0)].sort_values("max_conf", ascending=False).head(8)
        gallery(d, f"most confident false boxes on {c}", out / f"gallery_false_{c}.png")

    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved report.txt, image_level.csv, box_level.csv and galleries in {out}")


if __name__ == "__main__":
    main()
