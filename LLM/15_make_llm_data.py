"""
Step 15 — Training data for the LLM (Phase 3)
============================================

Runs anywhere (standard library + numpy + pandas). Needs step 14's outputs.

What one training example looks like
    prompt   = the findings record as short text + an instruction ("Write the report." or a question)
    response = the reference text written by llm_utils (templates with several wordings)
GPT-Neo learns prompt -> response. It never sees the image, only the findings.

Where the findings records come from
  1. REAL records: findings_val.jsonl from step 14 (every validation X-ray, built from the
     real classifier and detector predictions).
  2. SYNTHETIC records (default 3,000): random but realistic inputs pushed through the SAME
     rules (findings_utils.build_findings), so every record is one the app could produce.
     They cover cases that are rare in the real data: children, unknown view, borderline
     results, three findings, "review suggested", no localised region.
  3. TEST records: findings_test.jsonl (if step 14 --split test has been run). Kept OUT of
     training; step 17 uses them to evaluate the LLM on X-rays it has never seen.

Per record: 1 report + 3 question/answer pairs (questions drawn from 9 kinds, including
out-of-scope questions such as "Is this COVID-19?", which must be declined).
Records (not examples) are split 90 / 10 into llm_train / llm_val, so the same X-ray never
appears in both.

Output: <outputs>/llm_data/  llm_train.jsonl, llm_val.jsonl, llm_test.jsonl, summary.txt
Run (inside WSL, from the code folder):
    python 15_make_llm_data.py --run "$HOME/AIML Project/Test 2/RSNA/outputs/detector_runs/yolo11s_512_20261028-0921"
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml

import llm_utils as L
from findings_utils import build_findings, load_config

BAND_CUTS = [(0.1, "very_unlikely"), (0.3, "unlikely"), (0.7, "uncertain"), (0.9, "likely"), (1.01, "very_likely")]


def band(p):
    return next(name for cut, name in BAND_CUTS if p < cut)


def synthetic(cfg, rng):
    """One random but realistic input, pushed through the real rules."""
    view = rng.choices(["AP", "PA", "unknown"], [0.45, 0.45, 0.10])[0]
    age = None if rng.random() < 0.03 else (rng.randint(2, 17) if rng.random() < 0.12 else rng.randint(18, 92))
    sex = rng.choices(["F", "M", None], [0.47, 0.50, 0.03])[0]
    pos_like = rng.random() < 0.55
    p = float(np.clip(rng.betavariate(5, 1.2) if pos_like else rng.betavariate(1.2, 6), 0.001, 0.999))
    decision = int(p >= 0.58)
    if rng.random() < 0.06:                       # near-threshold disagreements (borderline cases)
        decision = 1 - decision
    clf = {"prob": p, "decision": decision, "band": band(p), "view": view}
    boxes = []
    n = rng.choices([0, 1, 2, 3], [0.05, 0.45, 0.42, 0.08])[0] if decision else rng.choices([0, 1], [0.93, 0.07])[0]
    top, bot = cfg["lung_top"], cfg["lung_bottom"]
    sides = rng.sample([0.08, 0.52], k=min(n, 2)) + [rng.choice([0.08, 0.52])] * max(0, n - 2)
    for x0 in sides:
        w = rng.uniform(0.08, 0.32); h = rng.uniform(0.08, 0.45)
        x1 = x0 + rng.uniform(0, 0.38 - w if w < 0.38 else 0)
        y1 = rng.uniform(top, max(top, bot - h))
        boxes.append([x1, y1, min(x1 + w, 0.99), min(y1 + h, 0.99), rng.uniform(0.05, 0.6)])
    return build_findings(cfg, clf, np.array(boxes).reshape(-1, 5), age=age, sex=sex)


def examples(rec, rid, source, rng):
    # the record travels with every example, so steps 16-17 can check generated text against it
    out = [{"id": rid, "source": source, "kind": "report", "prompt": L.prompt(rec), "response": L.write_report(rec, rng),
            "record": rec}]
    for q, a, k in L.make_qa(rec, rng, 3):
        out.append({"id": rid, "source": source, "kind": k, "prompt": L.prompt(rec, q), "response": a, "record": rec})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="detector run folder (steps 12-14)")
    ap.add_argument("--n-synthetic", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    outputs = Path(yaml.safe_load((args.run / "args.yaml").read_text())["data"]).parent.parent
    out = args.out or outputs / "llm_data"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.run / "findings_config.json")

    real = [json.loads(l) for l in open(args.run / "findings_val" / "findings_val.jsonl")]
    recs = [(f"val_{r['patientId']}", "real", r["record"]) for r in real]
    recs += [(f"syn_{i:05d}", "synthetic", synthetic(cfg, rng)) for i in range(args.n_synthetic)]
    rng.shuffle(recs)
    n_val = len(recs) // 10
    split = {"llm_val": recs[:n_val], "llm_train": recs[n_val:]}

    test_f = args.run / "findings_test" / "findings_test.jsonl"
    if test_f.exists():
        split["llm_test"] = [(f"test_{r['patientId']}", "real_test", r["record"]) for r in map(json.loads, open(test_f))]

    lines = []
    for name, rs in split.items():
        ex = [e for rid, src, rec in rs for e in examples(rec, rid, src, rng)]
        with open(out / f"{name}.jsonl", "w") as fh:
            for e in ex:
                fh.write(json.dumps(e) + "\n")
        chars = np.array([len(e["prompt"]) + len(e["response"]) for e in ex])
        bad = sum(bool(L.check(json.loads(json.dumps(rec)), L.write_report(rec, random.Random(1)))) for _, _, rec in rs)
        pos = sum(rec["classifier"]["result"] == "pneumonia" for _, _, rec in rs)
        lines.append(f"{name:10s} records {len(rs):5d} (pneumonia {pos}, real {sum(s != 'synthetic' for _, s, _ in rs)}) | "
                     f"examples {len(ex):6d} | length ~{int(chars.mean() / 4)} tokens on average, "
                     f"max ~{int(chars.max() / 4)} | reference reports failing the checker: {bad}")
    lines.append("llm_test is only for step 17 (never trained on)." if "llm_test" in split
                 else "llm_test: not written (run 14_findings.py --split test first).")
    print("\n".join(lines))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved in {out}")


if __name__ == "__main__":
    main()
