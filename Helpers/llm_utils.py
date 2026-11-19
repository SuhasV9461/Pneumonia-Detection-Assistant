"""
llm_utils.py — everything the LLM part shares: prompt format, reference reports,
questions and answers, and the faithfulness check
================================================================================

Used by step 15 (training data), step 16 (training), step 17 (evaluation) and the app.
Needs only the Python standard library.

Why reference texts are written by templates
  RSNA has labels and boxes, but NO written radiology reports. So the training targets are
  produced here from the findings record (step 14), with several wordings for every
  sentence. GPT-Neo learns to turn a findings record into a fluent report and to answer
  questions about it. Every fact it may state is in the record; the faithfulness check
  (check()) catches any sentence that states something the record does not contain.
  The LLM never sees the image.

Prompt format (one training example = prompt + target + end-of-text token)
    ### Findings
    age: 67 | sex: female | view: AP | result: pneumonia | probability: 0.99 (very likely)
    localisation: boxes | finding 1: right lung, upper and middle zones, large, clearly marked
    bilateral: yes | notes: none
    ### Instruction
    Write the report.            (or:  Question: Which lung is affected?)
    ### Response
    <text>
"""

import random
import re

END = "<|endoftext|>"
SEX = {"F": "female", "M": "male", None: "unknown"}


# ---------------------------------------------------------------------------- prompt
def _zones(f):
    z = f.get("zones") or [f["zone"]]
    return z[0] + " zone" if len(z) == 1 else " and ".join(z) + " zones"


def serialize(rec):
    p, c = rec["patient"], rec["classifier"]
    lines = [f"age: {p['age'] if p['age'] is not None else 'unknown'} | sex: {SEX.get(p['sex'], 'unknown')} | "
             f"view: {p['view']} | result: {c['result']} | probability: {c['probability']:.2f} ({c['band']})",
             f"localisation: {rec['localisation']}"]
    for i, f in enumerate(rec["findings"], 1):
        lines.append(f"finding {i}: {f['side']} lung, {_zones(f)}, {f['extent']}, {f['box_confidence']}")
    lines.append(f"bilateral: {'yes' if rec['bilateral'] else 'no'} | notes: "
                 + ("; ".join(rec["notes"]) if rec["notes"] else "none"))
    return "\n".join(lines)


def prompt(rec, question=None):
    task = "Write the report." if question is None else f"Question: {question}"
    return f"### Findings\n{serialize(rec)}\n### Instruction\n{task}\n### Response\n"


# ---------------------------------------------------------------------------- reference report
def _pick(rng, options):
    return rng.choice(options)


def _who(rec):
    p = rec["patient"]
    sex = SEX.get(p["sex"], "unknown")
    if p["age"] is None:
        return f"{sex} patient of unknown age" if sex != "unknown" else "patient of unknown age and sex"
    return f"{p['age']}-year-old {sex if sex != 'unknown' else 'patient'}"


def _borderline(rec):
    return any(n.startswith("Borderline result") for n in rec["notes"])


def _impression(rec, rng):
    c = rec["classifier"]; pr = f"{c['probability']:.2f}"
    if _borderline(rec):
        if c["result"] == "pneumonia":
            return _pick(rng, [f"The model flags a possible pneumonia, but the result is borderline (probability {pr}).",
                               f"Borderline result: the model flags a possible pneumonia (probability {pr})."])
        return _pick(rng, [f"No pneumonia-type opacity was confidently detected, but the result is borderline (probability {pr}).",
                           f"Borderline result: no pneumonia-type opacity was confidently detected (probability {pr})."])
    if c["result"] == "pneumonia":
        return _pick(rng, [
            f"Pneumonia is {c['band']} (probability {pr}).",
            f"The findings are consistent with pneumonia, rated {c['band']} (probability {pr}).",
            f"The model detects a pneumonia-type opacity; pneumonia is {c['band']} (probability {pr}).",
        ])
    return _pick(rng, [
        f"No pneumonia-type opacity was detected; pneumonia is {c['band']} (probability {pr}).",
        f"No radiographic evidence of pneumonia; pneumonia is {c['band']} (probability {pr}).",
        f"The X-ray does not show a pneumonia-type opacity; pneumonia is {c['band']} (probability {pr}).",
    ])


def _finding_sentence(f, rng, approximate_only=False):
    where = f"{f['side']} lung, {_zones(f)}"
    if approximate_only:
        return _pick(rng, [
            f"The exact location is uncertain; the most likely area is the {where}.",
            f"No clearly marked region was found; the most likely location is the {where}.",
        ])
    how = "clearly marked" if f["box_confidence"] == "clearly marked" else "marked approximately"
    return _pick(rng, [
        f"A {f['extent']} opacity is {how} in the {where}.",
        f"There is a {f['extent']} area of opacity in the {where}, {how}.",
        f"In the {where}, a {f['extent']} opacity is {how}.",
    ])


NOTE_TEXT = {
    "No localised region": ["No localised region was identified; the change may be diffuse or subtle."],
    "Small opacities on PA": ["Small opacities on PA views are the hardest case for this model."],
    "The location model marked": ["The location model marked a region although the overall result is negative; "
                                  "a review of the image is suggested."],
    "High probability below": ["The probability is high although it is below the decision threshold; "
                               "a review of the image is suggested."],
    "under 18": ["The patient is under 18, and the model is less reliable for children."],
    "View (AP / PA) unknown": ["The view was not recorded, so a general calibration was used."],
}


def _note_sentences(rec):
    out = []
    for n in rec["notes"]:
        for key, txt in NOTE_TEXT.items():
            if key in n:
                out.append(txt[0])
    return out


def write_report(rec, rng=None):
    rng = rng or random.Random(0)
    p, c = rec["patient"], rec["classifier"]
    parts = [_pick(rng, [f"Chest X-ray, {p['view']} view, {_who(rec)}.",
                         f"{p['view']} chest radiograph of a {_who(rec)}."]) if p["view"] != "unknown"
             else f"Chest X-ray, view not recorded, {_who(rec)}."]
    parts.append("Impression: " + _impression(rec, rng))
    if c["result"] == "pneumonia":
        if rec["localisation"] == "possible_location":
            parts.append(_finding_sentence(rec["findings"][0], rng, approximate_only=True))
        else:
            for f in rec["findings"]:
                parts.append(_finding_sentence(f, rng))
            if rec["bilateral"]:
                parts.append(_pick(rng, ["Both lungs are involved.", "The opacities involve both lungs."]))
    parts += _note_sentences(rec)
    parts.append(_pick(rng, ["Review by a radiologist and clinical correlation are recommended.",
                             "Clinical correlation and radiologist review are recommended."])
                 if c["result"] == "pneumonia" else
                 _pick(rng, ["If symptoms persist, clinical review is advised.",
                             "Clinical review is advised if symptoms persist."]))
    parts.append(rec["disclaimer"])
    return " ".join(parts)


# ---------------------------------------------------------------------------- questions and answers
def _where_answer(rec):
    c = rec["classifier"]
    if c["result"] != "pneumonia":
        return "No pneumonia-type opacity was detected, so no region is marked."
    if rec["localisation"] == "none":
        return "No localised region was identified; the change may be diffuse or subtle."
    fs = rec["findings"]
    if rec["localisation"] == "possible_location":
        return f"The exact location is uncertain; the most likely area is the {fs[0]['side']} lung, {_zones(fs[0])}."
    s = "; ".join(f"{f['side']} lung, {_zones(f)}" for f in fs)
    return f"The opacity is in the {s}." + (" Both lungs are involved." if rec["bilateral"] else "")


QUESTIONS = {
    "result": ["Is there pneumonia?", "Does this X-ray show pneumonia?", "Do I have pneumonia?",
               "What is the result?", "Is the X-ray normal?"],
    "where": ["Where is the pneumonia?", "Which lung is affected?", "Where is the opacity?",
              "Which part of the lung is involved?"],
    "extent": ["How large is it?", "How extensive is the opacity?", "Is it a big area?"],
    "bilateral": ["Are both lungs affected?", "Is it in both lungs?"],
    "confidence": ["How sure is the model?", "How confident is this result?", "What does the probability mean?"],
    "view": ["What kind of X-ray is this?", "Which view was used?"],
    "next": ["What should I do next?", "What do you recommend?"],
    "diagnosis": ["Is this a diagnosis?", "Can I rely on this result alone?"],
    "scope": ["Is this COVID-19?", "Can you see tuberculosis?", "Is there lung cancer?",
              "Can you tell which bacteria caused it?"],
}


def answer(rec, kind):
    c, p = rec["classifier"], rec["patient"]
    pr = f"{c['probability']:.2f}"
    pos = c["result"] == "pneumonia"
    if kind == "result" and any(n.startswith("Borderline result") for n in rec["notes"]):
        return (f"The model flags a possible pneumonia, but the result is borderline (probability {pr}); "
                "a review of the image is suggested." if pos else
                f"No pneumonia-type opacity was confidently detected, but the result is borderline (probability {pr}); "
                "a review of the image is suggested.")
    if kind == "result":
        return (f"The model found a pneumonia-type opacity: pneumonia is {c['band']} (probability {pr})."
                if pos else f"No pneumonia-type opacity was detected: pneumonia is {c['band']} (probability {pr}).")
    if kind == "where":
        return _where_answer(rec)
    if kind == "extent":
        if not pos or not rec["findings"]:
            return _where_answer(rec)
        return "; ".join(f"the {f['side']} lung opacity is {f['extent']}" for f in rec["findings"]).capitalize() + "."
    if kind == "bilateral":
        if not pos:
            return "No pneumonia-type opacity was detected in either lung."
        return "Yes, both lungs are involved." if rec["bilateral"] else (
            "No, only one lung is marked." if rec["findings"] else _where_answer(rec))
    if kind == "confidence":
        return (f"The calibrated probability is {pr}, which is in the {c['band']} range. On validation data, "
                f"results in a range like this were correct about as often as the number suggests.")
    if kind == "view":
        return (f"This is a {p['view']} chest X-ray." if p["view"] != "unknown"
                else "The view was not recorded, so a general calibration was used.")
    if kind == "next":
        return ("Please have the X-ray reviewed by a radiologist and discuss the result with a doctor."
                if pos else "If symptoms persist, please see a doctor for clinical review.")
    if kind == "diagnosis":
        return rec["disclaimer"]
    if kind == "scope":
        return ("This model only detects pneumonia-type lung opacity. It cannot identify the cause or other "
                "diseases; a doctor and further tests are needed for that.")
    raise ValueError(kind)


def make_qa(rec, rng, n=3):
    kinds = list(QUESTIONS)
    if rec["classifier"]["result"] != "pneumonia":
        kinds = [k for k in kinds if k not in ("extent",)]
    out = []
    for k in rng.sample(kinds, min(n, len(kinds))):
        out.append((rng.choice(QUESTIONS[k]), answer(rec, k), k))
    return out


# ---------------------------------------------------------------------------- faithfulness check
NEG = re.compile(r"\bno (pneumonia|radiographic evidence)|does not show a pneumonia|no pneumonia-type", re.I)
POS = re.compile(r"consistent with pneumonia|detects a pneumonia-type|found a pneumonia-type|flags a possible pneumonia"
                 r"|pneumonia is (very likely|likely)\b", re.I)


def facts_in_text(text):
    t = text.lower()
    sides = {s for s in ("right", "left") if re.search(rf"\b{s} lung", t)}
    zones = {z for z in ("upper", "middle", "lower") if re.search(rf"\b{z}\b", t)}
    extents = {e for e in ("small", "moderate", "large") if re.search(rf"\b{e} (opacity|area)|is {e}\b", t)}
    probs = {float(x) for x in re.findall(r"probability (?:is )?(0\.\d\d|1\.00)", t)}
    return {"negative": bool(NEG.search(t)), "positive": bool(POS.search(t)) and not NEG.search(t),
            "sides": sides, "zones": zones, "extents": extents, "probs": probs,
            "both": "both lungs" in t and "either lung" not in t}


def check(rec, text, kind="report"):
    """List of problems: facts in the text that the record does not support (hallucinations),
    and, for full reports, required facts that are missing (omissions)."""
    f = facts_in_text(text)
    pos = rec["classifier"]["result"] == "pneumonia"
    rs = {x["side"] for x in rec["findings"]}
    rz = {z for x in rec["findings"] for z in (x.get("zones") or [x["zone"]])}
    re_ = {x["extent"] for x in rec["findings"]}
    bad = []
    if pos and f["negative"]:
        bad.append("says no pneumonia, record says pneumonia")
    if not pos and f["positive"]:
        bad.append("says pneumonia, record says no pneumonia")
    if f["sides"] - rs:
        bad.append(f"invented side {sorted(f['sides'] - rs)}")
    if f["zones"] - rz:
        bad.append(f"invented zone {sorted(f['zones'] - rz)}")
    if f["extents"] - re_:
        bad.append(f"invented extent {sorted(f['extents'] - re_)}")
    if f["both"] and not rec["bilateral"] and pos:
        bad.append("says both lungs, record says one")
    if f["probs"] and round(rec["classifier"]["probability"], 2) not in f["probs"]:
        bad.append(f"wrong probability {sorted(f['probs'])}")
    if kind == "report":
        if not (f["negative"] or f["positive"]):
            bad.append("missing: result")
        if pos and rec["localisation"] != "none" and rs - f["sides"]:
            bad.append(f"missing side {sorted(rs - f['sides'])}")
        if rec["disclaimer"].lower()[:40] not in text.lower():
            bad.append("missing: disclaimer")
    return bad


def check_answer(rec, kind, text):
    """Problems in an answer: invented facts (as check()) PLUS the fact the question asks for
    must be present. A fluent but empty answer counts as wrong."""
    bad = check(rec, text, "qa")
    f = facts_in_text(text); t = text.lower()
    pos = rec["classifier"]["result"] == "pneumonia"
    sides = {x["side"] for x in rec["findings"]}
    neg_cue = f["negative"] or "no region" in t or "either lung" in t or "not confidently" in t
    need = {
        "result": (f["positive"] if pos else neg_cue) and bool(f["probs"]),
        "where": (bool(sides & f["sides"]) or (rec["localisation"] == "none" and "no localised" in t)) if pos else neg_cue,
        "extent": bool({x["extent"] for x in rec["findings"]} & set(re.findall(r"\b(small|moderate|large)\b", t)))
                  if pos and rec["findings"] else True,
        "bilateral": (t.startswith("yes") if rec["bilateral"] else (t.startswith("no") or neg_cue)) if pos else neg_cue,
        "confidence": bool(f["probs"]),
        "view": (rec["patient"]["view"].lower() in t) if rec["patient"]["view"] != "unknown" else "not recorded" in t,
        "next": any(w in t for w in ("doctor", "radiologist", "clinical")),
        "diagnosis": "not a diagnosis" in t,
        "scope": "only detects" in t,
    }
    if not need.get(kind, True):
        bad.append(f"missing: the fact the question asks for ({kind})")
    return bad
