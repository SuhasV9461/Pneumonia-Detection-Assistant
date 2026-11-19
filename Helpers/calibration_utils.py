"""
calibration_utils.py — turn a model's logit into an honest probability and a yes/no decision
============================================================================================

Written by step 4b (04b_calibrate_per_view.py) and used by everything after it (final
evaluation, the app, the chatbot), so the rules live in ONE place.

    from calibration_utils import load, probability, decide, band

    p    = probability(cal, logits, views)   # honest probability, for the report / chatbot
    flag = decide(cal, logits, views)        # yes / no, using the chosen decision policy

Why per view
  The classifier was trained with group weights, so it treats AP and PA X-rays as if they had
  the same pneumonia rate. That keeps its RANKING fair, but its probabilities ignore the real
  difference: in RSNA about 74 % of AP images and 16 % of PA images show opacity. Platt
  scaling fitted separately for AP and PA puts that difference back into the probability:

      probability = sigmoid(a_view * logit + b_view)

  a_view stretches or squeezes the scores (like a temperature), b_view shifts them (which a
  temperature cannot do). Two numbers per view, learned on the validation set.

Why the decision is a separate question
  With view-honest probabilities, a single cut-off would again treat PA X-rays as "probably
  normal" and miss PA pneumonia. So the yes/no decision has its own policy (see decide()):
  one threshold per view (same sensitivity in AP and PA), or one threshold on the view-blind
  score. Step 4b measures both and stores the chosen one.

If the view is unknown (e.g. a PNG uploaded without a DICOM header), the overall Platt
scaling and the single overall threshold are used instead.
"""

import json
from pathlib import Path

import numpy as np

VIEWS = ("AP", "PA")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype="float64")))


def load(path):
    """Read calibration_v2.json (written by step 4b)."""
    return json.loads(Path(path).read_text())


def _view_key(cal, view):
    return view if view in cal["params"] else "overall"


def probability(cal, logits, views):
    """Calibrated probability for each image (views: 'AP', 'PA' or anything else = unknown)."""
    logits = np.atleast_1d(np.asarray(logits, dtype="float64"))
    views = np.atleast_1d(np.asarray(views, dtype=object))
    out = np.empty(len(logits))
    for i, (z, v) in enumerate(zip(logits, views)):
        p = cal["params"][_view_key(cal, v)]
        out[i] = sigmoid(p["a"] * z + p["b"])
    return out


def decide(cal, logits, views):
    """1 = flag as pneumonia, 0 = no pneumonia-type opacity.

    Three policies (chosen in step 4b and stored in calibration_v2.json):
      per_view : view-calibrated probability >= the threshold of the image's view
                 (catches the same share of pneumonia in AP and in PA)
      neutral  : the VIEW-BLIND probability (overall Platt) >= one threshold
                 (uses the model's own view-balanced ranking, as trained with group weights)
      single   : view-calibrated probability >= one threshold
    """
    logits = np.atleast_1d(np.asarray(logits, dtype="float64"))
    views = np.atleast_1d(np.asarray(views, dtype=object))
    d = cal["decision"]
    th = d["thresholds"]
    if d["policy"] == "neutral":
        o = cal["params"]["overall"]
        return (sigmoid(o["a"] * logits + o["b"]) >= th["overall"]).astype(int)
    p = probability(cal, logits, views)
    if d["policy"] == "per_view":
        return np.array([int(pi >= th.get(v, th["overall"])) for pi, v in zip(p, views)])
    return (p >= th["overall"]).astype(int)


def band(cal, p):
    """Word for a calibrated probability (same cut-offs for every view)."""
    for b in cal["bands"]:
        if b["low"] <= p < b["high"]:
            return b["name"]
    return cal["bands"][-1]["name"]
