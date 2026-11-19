"""
Step 14 — The findings layer: classifier + detector -> one JSON per X-ray
=========================================================================

Runs in yolo-env OR ai-env (numpy / pandas only; no model is run here). It reuses the
predictions already saved by step 10 (classifier) and step 13 (detector), and applies the
rules in findings_utils.py. The JSON it writes is exactly what the LLM will read.

STAGE 1  --split val   (as often as you like)
  1. Fixes the geometry from the TRAINING boxes (no peeking at val):
       lung field = from the 5th percentile of box top edges to the 95th percentile of
       box bottom edges -> zones are thirds of that band, not of the whole image;
       extent cut-offs = thirds of all training box areas (small / moderate / large).
  2. Chooses the "review suggested" cut-off: the lowest detector confidence at which at
     most 1 % of NORMAL X-rays that the classifier calls negative would get the note.
  3. Checks the rules on validation:
       - for pneumonia the classifier finds: how often boxes / a possible location / nothing
         is reported, and how often the reported lung, lung + zone, and box are correct
       - box precision for each confidence word (clear / moderate / low)
       - pneumonia in both lungs: how often "bilateral" is reported
       - false alarms (normal called pneumonia): what the record says about them
  4. Freezes everything in findings_config.json and writes findings_val.jsonl.

STAGE 2  --split test  (ONCE, after step 13 --split test)
  Same checks with the frozen rules, plus the Not-Normal X-rays.

Output: <detector run>/findings_<split>/  findings_<split>.jsonl, report.txt, examples.json
        <detector run>/findings_config.json

Run (inside WSL, from the code folder; yolo-env is fine):
    python 14_findings.py --run "$HOME/AIML Project/Test 2/RSNA/outputs/detector_runs/yolo11s_512_20261028-0921"
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import calibration_utils as cu
from findings_utils import _iou, build_findings, side_of, zone_of

MAX_REVIEW_NORMAL = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="detector run folder (steps 12-13)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--yolo-data", type=Path, default=None, help="override the yolo_data folder")
    ap.add_argument("--final-model", type=Path, default=None, help="override <outputs>/final_model")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run = args.run
    data_dir = args.yolo_data or Path(yaml.safe_load((run / "args.yaml").read_text())["data"]).parent
    outputs = data_dir.parent
    fm = args.final_model or outputs / "final_model"
    out = run / f"findings_{args.split}"
    out.mkdir(exist_ok=True)
    cfg_f = run / "findings_config.json"
    if args.split == "test":
        if not cfg_f.exists():
            raise SystemExit("Run --split val first (the rules must be frozen on validation).")
        if (out / "report.txt").exists() and not args.force:
            raise SystemExit("The findings test evaluation has already been run once.")

    det_cfg = json.loads((run / "detector_config.json").read_text())
    cal = cu.load(fm / "calibration_v2.json")
    man = pd.read_csv(data_dir / "manifest.csv")
    man = man[man["split"] == args.split]
    boxes = pd.read_csv(outputs / "boxes.csv")
    raw = pd.read_csv(run / f"eval_{args.split}" / f"predictions_{args.split}.csv")
    clf = pd.read_csv(fm / f"{args.split}_predictions.csv")[["patientId", "logit"]]
    D = man.merge(clf, on="patientId")                         # images the classifier scored
    v = D["view"].values.astype(object)
    D["prob"] = cu.probability(cal, D["logit"].values, v)
    D["decision"] = cu.decide(cal, D["logit"].values, v)
    D["band"] = [cu.band(cal, p) for p in D["prob"]]
    R = {p: g[["x1", "y1", "x2", "y2", "conf"]].values for p, g in raw.groupby("patientId")}
    G = {}
    for p, g in boxes[boxes["split"] == args.split].groupby("patientId"):
        G[p] = np.c_[g.yolo_xc - g.yolo_w / 2, g.yolo_yc - g.yolo_h / 2, g.yolo_xc + g.yolo_w / 2, g.yolo_yc + g.yolo_h / 2]

    lines = []
    def say(s=""):
        print(s); lines.append(s)
    say(f"FINDINGS LAYER - {args.split} - {run.name} - {datetime.now().isoformat(timespec='seconds')}")
    say(f"images scored by the classifier: {len(D)} (" + ", ".join(f"{k} {n}" for k, n in D['class'].value_counts().items()) + ")")

    # ------------------------------------------------------------ 1-2. geometry + review cut-off
    if args.split == "val":
        tr = boxes[boxes["split"] == "train"]
        top = float((tr.yolo_yc - tr.yolo_h / 2).quantile(0.05))
        bottom = float((tr.yolo_yc + tr.yolo_h / 2).quantile(0.95))
        area = 100 * tr.yolo_w * tr.yolo_h
        cfg = {"box_conf": det_cfg["conf"], "merge_iou": det_cfg["merge_iou"], "fallback_conf": 0.05,
               "clear_conf": 0.35, "lung_top": round(top, 4), "lung_bottom": round(bottom, 4),
               "extent_cuts": [round(float(area.quantile(1 / 3)), 2), round(float(area.quantile(2 / 3)), 2)]}
        say(); say("=" * 88); say("1. GEOMETRY from the training boxes"); say("=" * 88)
        say(f"lung field: {top:.3f} to {bottom:.3f} of the image height -> zones are thirds of this band")
        say(f"extent: small < {cfg['extent_cuts'][0]} % of the image <= moderate < {cfg['extent_cuts'][1]} % <= large")

        neg = D[D["decision"] == 0]
        best = np.array([R[p][:, 4].max() if p in R else 0.0 for p in neg["patientId"]])
        say(); say("=" * 88)
        say("2. 'REVIEW SUGGESTED' cut-off (classifier says NO, detector still marks a region)"); say("=" * 88)
        say(f"{'cut-off':>8s}{'normal with note':>18s}{'missed pneumonia with note':>28s}")
        chosen = None
        for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            rn = (best[neg["class"].values == "normal"] >= t).mean()
            ro = (best[neg["class"].values == "opacity"] >= t)
            say(f"{t:8.2f}{rn:18.3f}{f'{ro.mean():.3f} ({int(ro.sum())} of {len(ro)})':>28s}")
            if chosen is None and rn <= MAX_REVIEW_NORMAL:
                chosen = t
        cfg["review_conf"] = chosen if chosen is not None else 0.50
        say(f"CHOSEN: {cfg['review_conf']:.2f} (lowest cut-off with at most {100 * MAX_REVIEW_NORMAL:.0f} % of normals noted)")
        cfg["frozen_at"] = datetime.now().isoformat(timespec="seconds")
        cfg_f.write_text(json.dumps(cfg, indent=2))
    cfg = json.loads(cfg_f.read_text())

    # ------------------------------------------------------------ 3. build + check
    recs, rows, brows = [], [], []
    for r in D.to_dict("records"):
        clf_d = {"prob": r["prob"], "decision": int(r["decision"]), "band": r["band"], "view": r["view"]}
        rec = build_findings(cfg, clf_d, R.get(r["patientId"], np.zeros((0, 5))), age=r["age"], sex=r["sex"])
        recs.append({"patientId": r["patientId"], "class": r["class"], "split": args.split, "record": rec})
        gt = G.get(r["patientId"], np.zeros((0, 4)))
        gs = {side_of(g) for g in gt}; gz = {(side_of(g), zone_of(g, cfg)) for g in gt}
        for f in rec["findings"]:
            b = np.array([f["box"]])
            brows.append({"class": r["class"], "words": f["box_confidence"],
                          "iou_hit": bool(len(gt) and _iou(b, gt).max() >= 0.4),
                          "side_ok": f["side"] in gs, "side_zone_ok": (f["side"], f["zone"]) in gz})
        rows.append({"class": r["class"], "view": r["view"], "positive": rec["classifier"]["result"] == "pneumonia",
                     "localisation": rec["localisation"], "n_find": len(rec["findings"]),
                     "bilateral_rep": rec["bilateral"], "bilateral_gt": len(gs) == 2,
                     "review": any("review suggested" in n for n in rec["notes"])})
    F = pd.DataFrame(rows); Bx = pd.DataFrame(brows)
    with open(out / f"findings_{args.split}.jsonl", "w") as fh:
        for x in recs:
            fh.write(json.dumps(x) + "\n")

    say(); say("=" * 88); say("3. WHAT THE RECORDS SAY"); say("=" * 88)
    for c in ("opacity", "normal", "not_normal"):
        d = F[F["class"] == c]
        if not len(d):
            continue
        p = d[d.positive]
        say(f"{c}: {len(d)} images, classifier positive {len(p)} ({100 * len(p) / len(d):.1f} %)")
        if len(p):
            lc = p["localisation"].value_counts(normalize=True)
            say("   of those: " + ", ".join(f"{k} {100 * lc.get(k, 0):.1f} %" for k in ("boxes", "possible_location", "none")))
            if c == "opacity":
                for vv in ("AP", "PA"):
                    q = p[p.view == vv]; lq = q["localisation"].value_counts(normalize=True)
                    say(f"   {vv} ({len(q)}): " + ", ".join(f"{k} {100 * lq.get(k, 0):.1f} %" for k in ("boxes", "possible_location", "none")))
        n_rev = int(d.review.sum())
        if n_rev:
            say(f"   'review suggested' notes: {n_rev} ({100 * n_rev / len(d):.1f} % of all {c})")
    say()
    o = Bx[Bx["class"] == "opacity"]
    if len(o):
        say(f"Reported findings on real pneumonia ({len(o)} findings):")
        say(f"{'box confidence word':26s}{'n':>6s}{'right lung':>12s}{'lung + zone':>13s}{'box overlap >= 0.4':>20s}")
        for w in ("clearly marked", "approximate"):
            d = o[o.words == w]
            if len(d):
                say(f"{w:26s}{len(d):6d}{d.side_ok.mean():12.3f}{d.side_zone_ok.mean():13.3f}{d.iou_hit.mean():20.3f}")
        say(f"{'all':26s}{len(o):6d}{o.side_ok.mean():12.3f}{o.side_zone_ok.mean():13.3f}{o.iou_hit.mean():20.3f}")
        pb = F[(F["class"] == "opacity") & F.positive]
        bg = pb[pb.bilateral_gt]
        say(f"Pneumonia in both lungs, classifier positive: {len(bg)}; reported as bilateral: {100 * bg.bilateral_rep.mean():.1f} %")
        zc = pd.Series([len(f["zones"]) for x in recs if x["class"] == "opacity" for f in x["record"]["findings"]])
        say(f"Findings spanning more than one zone: {100 * (zc > 1).mean():.1f} %")
        say(f"Reported as bilateral when the radiologist marked one lung only: "
            f"{100 * pb[~pb.bilateral_gt].bilateral_rep.mean():.1f} %")
    fa = Bx[Bx["class"] != "opacity"]
    say(f"Findings written on false alarms (classifier positive on a non-pneumonia X-ray): {len(fa)}")

    # examples
    ex = {}
    for name, cond in (("pneumonia_with_boxes", lambda x: x["record"]["localisation"] == "boxes" and x["class"] == "opacity"),
                       ("pneumonia_possible_location", lambda x: x["record"]["localisation"] == "possible_location"),
                       ("negative", lambda x: x["record"]["classifier"]["result"] == "no pneumonia" and not x["record"]["notes"]),
                       ("negative_review", lambda x: any("review" in n for n in x["record"]["notes"]))):
        hit = next((x for x in recs if cond(x)), None)
        if hit:
            ex[name] = hit
    (out / "examples.json").write_text(json.dumps(ex, indent=2))
    say(); say("Example record (pneumonia with boxes):")
    if "pneumonia_with_boxes" in ex:
        say(json.dumps(ex["pneumonia_with_boxes"]["record"], indent=2))
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved findings_{args.split}.jsonl, report.txt, examples.json in {out}")


if __name__ == "__main__":
    main()
