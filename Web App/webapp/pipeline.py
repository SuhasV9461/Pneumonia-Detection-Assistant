"""
pipeline.py — everything the web application does with ONE X-ray (Phase 4)
==========================================================================

Runs in yolo-env (PyTorch + onnxruntime + transformers + peft; no TensorFlow).
Reads the frozen models packaged by step 18 (<outputs>/webapp_models/).

    from pipeline import Pipeline
    pipe = Pipeline(models_dir)                       # loads everything once
    res = pipe.analyze(file_bytes, filename, age=None, sex=None)
    ans = pipe.chat(res["record"], "Which lung is affected?")

The steps, each copied exactly from the scripts that produced the frozen numbers
  1. read      DICOM (age, sex, view from the header) or PNG / JPG
               -> grey uint8 -> 512 x 512 with INTER_AREA                      (step 1)
  2. classify  512 -> 384 'area' resize -> 3 channels -> 5 TTA versions
               (original, rotate +-5 deg, zoom 1.06 / 0.94) -> ImageNet z-score
               -> both ONNX members -> mean logit                               (step 10)
               -> per-view Platt probability, neutral decision, band            (step 4b)
               heat-map (CAM) = ReLU(feature maps . Dense weights), both members (step 5)
  3. detect    YOLO on the 512 image, raw boxes >= 0.01                        (step 13)
  4. findings  build_findings with the frozen rules                             (step 14)
  5. report    GPT-Neo + LoRA writes the report from the findings ONLY          (step 16)
               SAFETY NET: the report is checked with llm_utils.check(); if it states
               anything the findings do not contain, or misses a required fact, the
               template report is shown instead and this is said on the page.
  6. chat      the same for answers (llm_utils.check_answer for invented facts).

Self-test (run this once after step 18; it must pass before the website is trusted):
    python pipeline.py --outputs "$HOME/AIML Project/Test 2/RSNA/outputs" --selftest 50
  re-analyses 50 validation PNGs and compares with what steps 10, 13 and 14 saved:
  classifier logits per member and TTA version, decisions, detector boxes, findings records.
"""

import io
import json
import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # the code folder: calibration_utils, findings_utils, llm_utils

import calibration_utils as cu                 # noqa: E402
import findings_utils as fu                    # noqa: E402
import llm_utils as L                          # noqa: E402

SEX_CODES = {"M": "M", "F": "F", "MALE": "M", "FEMALE": "F"}

# which of the 9 trained question kinds a free question belongs to (first match wins);
# None = not recognised -> only the "no invented facts" check applies
ROUTES = [
    ("scope", r"covid|corona|tuberculo|\btb\b|cancer|tumou?r|bacteri|virus|viral|fungal|cause|what disease|other disease"),
    ("diagnosis", r"diagnos|rely|trust this|replace .*doctor"),
    ("next", r"what should|next step|recommend|what (do|can) i do|treat|medicine|antibiotic|hospital|see a doctor"),
    ("bilateral", r"\bboth\b|two lungs|each lung|either lung"),
    ("extent", r"\bbig\b|large|size|extensive|how much|severe|spread"),
    ("where", r"where|which (lung|side|part|zone|area)|location|located|left|right|upper|lower|middle"),
    ("confidence", r"sure|confiden|probab|certain|accura|percent|how likely|chance"),
    ("view", r"\bview\b|kind of x-?ray|type of x-?ray|\bap\b|\bpa\b|projection"),
    ("result", r"pneumonia|normal|result|is there|do i have|positive|negative|infect|sick|healthy|clear"),
]


def route(question):
    q = question.lower()
    return next((k for k, pat in ROUTES if re.search(pat, q)), None)


# ---------------------------------------------------------------------------- 1. read
def read_image(data, filename):
    """Bytes of a DICOM / PNG / JPG -> (512 x 512 uint8 grey image, header dict)."""
    head = {}
    name = (filename or "").lower()
    is_dicom = name.endswith((".dcm", ".dicom")) or data[128:132] == b"DICM"
    if is_dicom:
        import pydicom
        d = pydicom.dcmread(io.BytesIO(data))
        img = d.pixel_array.astype(np.float64)
        if str(d.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            img = img.max() - img                                   # white = air -> invert
        if d.pixel_array.dtype != np.uint8 or str(d.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
        age = str(d.get("PatientAge", "")).strip().upper().rstrip("Y").lstrip("0")
        head = {"age": int(age) if age.isdigit() else None,
                "sex": SEX_CODES.get(str(d.get("PatientSex", "")).strip().upper()),
                "view": str(d.get("ViewPosition", "")).strip().upper() or None}
    else:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read the file: upload a DICOM, PNG or JPG chest X-ray.")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape != (512, 512):
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    return img, head


# ---------------------------------------------------------------------------- 2. classify
# The classifier was trained on images resized by tf.image.resize(method="area"). OpenCV's
# INTER_AREA gives the same average but breaks ties at .5 differently (~8,000 of 147,456 pixels
# differ by one grey level after rounding), which moved single logits by up to 0.5. So the exact
# TensorFlow algorithm is reproduced here: same weights, same float32 arithmetic, same order
# (checked: 0 pixels differ from TensorFlow).
def _axis(in_size, out_size):
    """Per output index: list of (input index, weight), as TensorFlow's ResizeArea computes them (float32)."""
    scale = np.float32(in_size) / np.float32(out_size)
    out = []
    for o in range(out_size):
        a = np.float32(o) * scale
        b = np.float32(o + 1) * scale
        terms = []
        for i in range(int(math.floor(a)), int(math.ceil(b))):
            if i < a:
                w = scale if (i + 1 > b) else np.float32(i + 1) - a
            else:
                w = (b - np.float32(i)) if (i + 1 > b) else np.float32(1.0)
            terms.append((min(max(i, 0), in_size - 1), np.float32(w)))
        out.append(terms)
    return out, scale

def tf_area_resize(img, size):
    """Exact copy of tf.image.resize(method='area') for one grey image (H, W) -> (size, size), float32."""
    img = img.astype(np.float32)
    H, W = img.shape
    ys, sy = _axis(H, size); xs, sx = _axis(W, size)
    k = max(len(t) for t in xs)
    # row sums over x: R[i, x] = sum_j value[i, j] * wx_j   (accumulated in TF's order)
    R = np.zeros((H, size), np.float32)
    for n in range(k):
        idx = np.array([t[n][0] if n < len(t) else 0 for t in xs]); w = np.array([t[n][1] if n < len(t) else 0 for t in xs], np.float32)
        R = (R + img[:, idx] * w).astype(np.float32)
    out = np.zeros((size, size), np.float32)
    k = max(len(t) for t in ys)
    for n in range(k):
        idx = np.array([t[n][0] if n < len(t) else 0 for t in ys]); w = np.array([t[n][1] if n < len(t) else 0 for t in ys], np.float32)
        out = (out + R[idx, :] * w[:, None]).astype(np.float32)
    return (out * np.float32(1.0 / (sy * sx))).astype(np.float32)


def tta_affine(img, angle_deg, zoom):
    """Same transform as step 10 (output pixel -> input pixel, centre w/2, black fill)."""
    if angle_deg == 0 and zoom == 1:
        return img
    h, w = img.shape[:2]
    a = math.radians(angle_deg)
    c, s = math.cos(a) / zoom, math.sin(a) / zoom
    cx, cy = w / 2.0, h / 2.0
    M = np.array([[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy]], dtype=np.float64)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class Classifier:
    def __init__(self, mdir, manifest):
        import onnxruntime as ort
        avail = ort.get_available_providers()
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.members = []
        for m in manifest["classifier"]["members"]:
            sess = ort.InferenceSession(str(mdir / m["onnx"]), providers=prov)
            cam = np.load(mdir / m["cam"])
            self.members.append((sess, cam["w"], m["img_size"]))
        self.tta = manifest["classifier"]["tta"]
        nm = manifest["classifier"]["normalisation"]
        self.mean, self.std = np.array(nm["mean"], np.float32), np.array(nm["std"], np.float32)

    def batch(self, img512, size):
        # TensorFlow 'area' resize of the 512 PNG, rounded back to uint8 (as tf.data did)
        small = np.round(tf_area_resize(img512, size))           # exactly as tf.image.resize(..., "area")
        small = np.clip(small, 0, 255).astype(np.float32)
        out = []
        for _, ang, zoom in self.tta:
            t = tta_affine(small, ang, zoom)
            rgb = np.repeat(t[..., None], 3, axis=2)
            out.append((rgb / 255.0 - self.mean) / self.std)
        return np.stack(out).astype(np.float32)

    def __call__(self, img512):
        logits, cams = [], []
        for sess, w, size in self.members:
            x = self.batch(img512, size)
            z, fmap = sess.run(None, {sess.get_inputs()[0].name: x})
            logits.append(z.ravel())                                   # one logit per TTA version
            cams.append(np.maximum(fmap[0] @ w, 0))                    # original version only
        cam = np.mean(cams, axis=0)
        cam = cv2.resize(cam, (512, 512), interpolation=cv2.INTER_CUBIC)
        cam = np.clip(cam / cam.max(), 0, 1) if cam.max() > 0 else cam
        return np.array(logits), cam                                    # (members, TTA), (512, 512)


# ---------------------------------------------------------------------------- the whole pipeline
class Pipeline:
    def __init__(self, models_dir, use_llm=True):
        self.dir = Path(models_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.cal = cu.load(self.dir / "calibration_v2.json")
        self.fcfg = fu.load_config(self.dir / "findings_config.json")
        self.classifier = Classifier(self.dir, self.manifest)
        from ultralytics import YOLO
        self.detector = YOLO(str(self.dir / self.manifest["detector"]["weights"]))
        self.use_llm = use_llm
        self._llm = None

    # ---- LLM, loaded on first use (it takes ~30 s and 2.6 GB of GPU memory)
    def llm(self):
        if self._llm is None and self.use_llm:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            tok = AutoTokenizer.from_pretrained(self.dir / self.manifest["llm"]["adapter"])
            base = AutoModelForCausalLM.from_pretrained(self.manifest["llm"]["base_model"],
                                                        torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
            model = PeftModel.from_pretrained(base, self.dir / self.manifest["llm"]["adapter"])
            model = model.merge_and_unload().to(dev).eval()            # fold LoRA into the weights: faster
            self._llm = (model, tok, dev)
        return self._llm

    def generate(self, prompt, max_new):
        import torch
        model, tok, dev = self.llm()
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, eos_token_id=tok.eos_token_id,
                                 pad_token_id=tok.eos_token_id, repetition_penalty=1.05)
        return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # ---- steps 2-4 without the LLM (also used by the self-test)
    def findings(self, img512, age=None, sex=None, view=None):
        logits, cam = self.classifier(img512)
        z = float(logits.mean())
        v = view if view in ("AP", "PA") else "unknown"
        prob = float(cu.probability(self.cal, [z], [v])[0])
        clf = {"prob": prob, "decision": int(cu.decide(self.cal, [z], [v])[0]), "band": cu.band(self.cal, prob), "view": v}
        r = self.detector.predict(cv2.cvtColor(img512, cv2.COLOR_GRAY2BGR), imgsz=self.manifest["detector"]["imgsz"],
                                  conf=0.01, iou=0.7, max_det=20, augment=False, verbose=False)[0]
        boxes = np.c_[r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy()] if len(r.boxes) else np.zeros((0, 5))
        rec = fu.build_findings(self.fcfg, clf, boxes, age=age, sex=sex)
        return rec, {"logits": logits, "cam": cam, "raw_boxes": boxes}

    # ---- the full analysis for the website
    def analyze(self, data, filename, age=None, sex=None, view=None):
        t0 = time.time()
        img, head = read_image(data, filename)
        age = age if age is not None else head.get("age")
        sex = sex or head.get("sex")
        view = view or head.get("view")
        rec, extra = self.findings(img, age, sex, view)
        t1 = time.time()
        report, source, problems = L.write_report(rec), "template", []
        if self.use_llm:
            gen = self.generate(L.prompt(rec), 260)
            problems = L.check(rec, gen, "report")
            report, source = (gen, "llm") if not problems else (report, "template (LLM text failed the check)")
        return {"record": rec, "report": report, "report_source": source, "llm_problems": problems,
                "image": img, "cam": extra["cam"], "header": head,
                "seconds": {"models": round(t1 - t0, 2), "report": round(time.time() - t1, 2)}}

    def chat(self, rec, question):
        """Answer a free question about THIS X-ray. The LLM writes the answer; it is shown only if
        (1) it states nothing the findings do not contain, (2) it contains the fact the question
        asks for (when the kind of question is recognised) and (3) it is real text. Otherwise
        the checked template answer for that kind of question is shown, and the page says so."""
        question = question.strip()[:300]
        kind = route(question)
        if not self.use_llm:
            return {"answer": L.answer(rec, kind or "result"), "source": "template", "kind": kind}
        ans = self.generate(L.prompt(rec, question), 120)
        problems = L.check_answer(rec, kind, ans) if kind else L.check(rec, ans, "qa")
        words = re.findall(r"[A-Za-z]{2,}", ans)
        if len(words) < 3 or sum(map(len, words)) < 0.6 * len(ans.replace(" ", "")):
            problems.append("not readable text")
        if problems:
            return {"answer": L.answer(rec, kind or "result"), "source": "fallback", "kind": kind, "problems": problems}
        return {"answer": ans, "source": "llm", "kind": kind}


# ---------------------------------------------------------------------------- self-test
def selftest(outputs, n):
    import pandas as pd
    mdir = outputs / "webapp_models"
    pipe = Pipeline(mdir, use_llm=False)
    meta = pd.read_csv(outputs / "metadata.csv")
    fm = outputs / "final_model"
    cache = np.load(fm / "cache_val.npz", allow_pickle=True)
    ids = list(cache["ids"])
    det_run = None
    for d in (outputs / "detector_runs").iterdir():
        if (d / "findings_val" / "findings_val.jsonl").exists() and (d / "findings_config.json").exists():
            det_run = d
    raw = pd.read_csv(det_run / "eval_val" / "predictions_val.csv")
    recs = {json.loads(l)["patientId"]: json.loads(l)["record"] for l in open(det_run / "findings_val" / "findings_val.jsonl")}
    val = meta[(meta.split == "val") & meta["class"].isin(["normal", "opacity"])].head(n)
    members = [m["source_run"] for m in pipe.manifest["classifier"]["members"]]
    dz, dbox, same_dec, same_rec, flips = [], [], 0, 0, []
    for r in val.itertuples(index=False):
        p = Path(r.png_path)
        p = p if p.is_absolute() else outputs / p
        img, _ = read_image(p.read_bytes(), p.name)
        age = None if pd.isna(r.age) else int(r.age)
        rec, extra = pipe.findings(img, age, r.sex, r.view)
        ref = np.stack([cache[m][ids.index(r.patientId)] for m in members])
        dz.append(float(np.abs(extra["logits"] - ref).max()))
        old = raw[raw.patientId == r.patientId].sort_values("conf", ascending=False)
        new = extra["raw_boxes"][np.argsort(-extra["raw_boxes"][:, 4])] if len(extra["raw_boxes"]) else np.zeros((0, 5))
        if len(old) and len(new):
            dbox.append(float(np.abs(old.iloc[0][["x1", "y1", "x2", "y2", "conf"]].values.astype(float) - new[0]).max()))
        ok_dec = rec["classifier"]["result"] == recs[r.patientId]["classifier"]["result"]
        same_dec += ok_dec
        if not ok_dec:
            flips.append(f"{r.patientId[:8]}: probability {rec['classifier']['probability']} vs "
                         f"{recs[r.patientId]['classifier']['probability']} (near the threshold)")
        strip = lambda x: json.dumps({k: v for k, v in x.items() if k != "findings"}, sort_keys=True)
        same_rec += strip(rec) == strip(recs[r.patientId])
    print(f"SELF-TEST on {len(val)} validation X-rays")
    print(f"  classifier logits vs step 10 (every member and TTA version): max difference {max(dz):.4f}, "
          f"median {np.median(dz):.4f}  (OK if < 0.10: float32 here vs mixed precision in step 10)")
    print(f"  best detector box vs step 13: max difference {max(dbox) if dbox else 0:.4f}  (OK if < 0.02)")
    print(f"  same decision as step 14: {same_dec}/{len(val)} | same record (except box coordinates): {same_rec}/{len(val)}"
          "  (records can differ in the 2nd decimal of the probability)")
    for f in flips:
        print("    decision changed: " + f)
    ok = max(dz) < 0.10 and (not dbox or max(dbox) < 0.02) and len(val) - same_dec <= max(1, len(val) // 100)
    print("  RESULT: " + ("PASS - the website reproduces the frozen pipeline" if ok else "CHECK - send me this output"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, required=True)
    ap.add_argument("--selftest", type=int, default=50)
    a = ap.parse_args()
    selftest(a.outputs, a.selftest)
