"""
findings_utils.py — turn the two models' outputs into ONE findings record (JSON)
===============================================================================

Used by step 14 (to check the rules on validation / test) and later by the app and the
LLM, so the rules live in one place. Needs only numpy (works in yolo-env and ai-env).

    from findings_utils import build_findings, load_config
    rec = build_findings(cfg, clf, boxes, age=63, sex="F")

  cfg   : findings_config.json (frozen by step 14) + calibration_v2.json bands
  clf   : {"prob": 0.93, "decision": 1, "band": "very_likely", "view": "PA"}   (from step 10)
  boxes : array (n, 5) = x1, y1, x2, y2, confidence, as fractions of the image, RAW from
          the detector (confidence >= 0.01). Merging and cut-offs are applied here.

The rules (each one checked on validation in step 14)
  1. The CLASSIFIER decides pneumonia yes / no. The detector never overrules it.
  2. If yes : report the detector boxes with confidence >= box_conf (0.20) after merging
              overlapping boxes (> merge_iou). If there are none, report the single best
              box with confidence >= fallback_conf as a "possible location".
              If there is no box at all: "no localised region" (diffuse or subtle change).
  3. If no  : report no boxes. If the detector still has a box >= review_conf, add a
              "review suggested" note (the two models disagree).
  4. Where  : side = which half of the image (the patient's RIGHT lung is on the image LEFT);
              zone = upper / middle / lower third of the LUNG FIELD (not of the image: the
              lungs only span about lung_top ... lung_bottom of the image height);
              extent = small / moderate / large from the box area (thirds of all
              radiologist boxes in the training split).
     zones  = every zone holding >= 30 % of the box height (a large box can span two).
     words  = "clearly marked" (confidence >= clear_conf) or "approximate".
  5. Notes  : age < 18, small opacity on a PA view, unknown view, and "borderline" when
              the decision and the probability band point different ways (e.g. an AP X-ray
              called negative with probability 0.75: "review suggested").
"""

import json
from pathlib import Path

import numpy as np

BAND_WORDS = {"very_unlikely": "very unlikely", "unlikely": "unlikely", "uncertain": "uncertain",
              "likely": "likely", "very_likely": "very likely"}
DISCLAIMER = ("This is an automated research tool, not a diagnosis. "
              "A qualified clinician must review the X-ray and the result.")


def load_config(path):
    return json.loads(Path(path).read_text())


def _iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-12)


def merge(boxes, thr):
    """Strongest box first; drop every box overlapping a kept box by more than thr."""
    if len(boxes) == 0:
        return boxes
    b = boxes[np.argsort(-boxes[:, 4])]
    keep = []
    for i in range(len(b)):
        if not keep or _iou(b[i:i + 1, :4], b[keep, :4]).max() <= thr:
            keep.append(i)
    return b[keep]


def side_of(box):
    return "right" if (box[0] + box[2]) / 2 < 0.5 else "left"     # patient's right = image left


def zone_of(box, cfg):
    yc = (box[1] + box[3]) / 2
    t = (yc - cfg["lung_top"]) / (cfg["lung_bottom"] - cfg["lung_top"])
    return "upper" if t < 1 / 3 else ("middle" if t < 2 / 3 else "lower")


def extent_of(box, cfg):
    a = 100 * (box[2] - box[0]) * (box[3] - box[1])
    return "small" if a < cfg["extent_cuts"][0] else ("moderate" if a < cfg["extent_cuts"][1] else "large")


def zones_covered(box, cfg, min_share=0.3):
    """Every zone that holds at least min_share of the box's height (a large box spans several)."""
    lo, hi = cfg["lung_top"], cfg["lung_bottom"]
    edges = [lo, lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3, hi]
    h = max(box[3] - box[1], 1e-6)
    out = []
    for name, a, b in zip(("upper", "middle", "lower"), edges[:-1], edges[1:]):
        if a == lo:
            a = 0.0                      # anything above the lung field counts as upper
        if b == hi:
            b = 1.0                      # anything below counts as lower
        if max(0.0, min(box[3], b) - max(box[1], a)) / h >= min_share:
            out.append(name)
    return out or [zone_of(box, cfg)]


def box_words(conf, cfg):
    """Two words only: on validation, 'approximate' boxes below the cut-off were as often on
    the right lung as boxes just above it, once the classifier had said pneumonia."""
    return "clearly marked" if conf >= cfg["clear_conf"] else "approximate"


def build_findings(cfg, clf, boxes, age=None, sex=None):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 5)
    view = clf.get("view") if clf.get("view") in ("AP", "PA") else "unknown"
    positive = bool(clf["decision"])
    b = merge(boxes[boxes[:, 4] >= cfg["fallback_conf"]], cfg["merge_iou"])

    findings, localisation, notes = [], "none", []
    if positive:
        strong = b[b[:, 4] >= cfg["box_conf"]]
        chosen = strong if len(strong) else b[:1]
        localisation = "boxes" if len(strong) else ("possible_location" if len(b) else "none")
        for bb in chosen:
            findings.append({"side": side_of(bb), "zone": zone_of(bb, cfg), "zones": zones_covered(bb, cfg),
                             "extent": extent_of(bb, cfg),
                             "box_confidence": box_words(bb[4], cfg),
                             "box": [round(float(x), 4) for x in bb[:4]], "score": round(float(bb[4]), 3)})
        if localisation == "none":
            notes.append("No localised region was found: the change may be diffuse or subtle.")
        if view == "PA" and findings and all(f["extent"] == "small" for f in findings):
            notes.append("Small opacities on PA views are the hardest case for this model.")
    elif len(b) and b[0, 4] >= cfg["review_conf"]:
        notes.append("The location model marked a region although the overall result is negative: "
                     "review suggested.")
    # The yes / no uses one view-blind threshold (Phase 1, neutral policy) while the probability
    # is view-specific, so the two can point in different directions near the threshold.
    band = clf["band"]
    if (band == "uncertain" or (positive and band in ("unlikely", "very_unlikely"))
            or (not positive and band in ("likely", "very_likely"))):
        notes.append("Borderline result: the probability is close to the decision threshold.")
    if not positive and band in ("likely", "very_likely"):
        notes.append("High probability below the decision threshold: review suggested.")
    if age is not None and age == age and age < 18:
        notes.append("Patient is under 18: the model is less reliable for children.")
    if view == "unknown":
        notes.append("View (AP / PA) unknown: a general calibration was used.")

    sides = sorted({f["side"] for f in findings})
    return {
        "patient": {"age": None if age is None or age != age else int(age), "sex": sex, "view": view},
        "classifier": {"result": "pneumonia" if positive else "no pneumonia",
                       "probability": round(min(max(float(clf["prob"]), 0.01), 0.99), 2),   # never "100 %"
                       "band": BAND_WORDS.get(clf["band"], clf["band"])},
        "localisation": localisation,
        "findings": findings,
        "bilateral": len(sides) == 2,
        "notes": notes,
        "disclaimer": DISCLAIMER,
    }
