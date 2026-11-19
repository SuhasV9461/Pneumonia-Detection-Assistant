"""
Step 17 — Evaluate the fine-tuned LLM on X-rays it has never seen (Phase 3)
==========================================================================

Run in yolo-env after step 16.

Input : an LLM run folder from step 16 (adapter/) + llm_test.jsonl from step 15
        (findings records built from the TEST X-rays; never used in training).
What it measures
  1. Reports: for N test records (default 300, stratified: pneumonia with boxes, possible
     location, no pneumonia, borderline, notes), the model writes a report with greedy
     decoding. Every report goes through the faithfulness check (llm_utils.check):
       hallucination = states something the record does not contain (wrong result, a lung,
                       zone or extent that is not in the findings, wrong probability,
                       "both lungs" when only one is marked)
       omission      = leaves out the result, an affected lung, or the disclaimer
  2. Questions: for the same records, 3 questions each (incl. out-of-scope questions such as
     "Is this COVID-19?"). Answers are checked for hallucinations; out-of-scope answers must
     decline ("only detects pneumonia-type lung opacity").
  3. The worst examples are saved so you can read them yourself.

Output: <llm run>/eval_test/  report.txt, generations.jsonl
Run (inside WSL, yolo-env active, from the code folder):
    python 17_eval_llm.py --run "$HOME/AIML Project/Test 2/RSNA/outputs/llm_runs/<run name>"
"""

import argparse
import collections
import json
import random
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import llm_utils as L


def group_of(rec):
    if any(n.startswith("Borderline") for n in rec["notes"]):
        return "borderline"
    if rec["classifier"]["result"] != "pneumonia":
        return "no pneumonia"
    return {"boxes": "pneumonia, boxes", "possible_location": "pneumonia, possible location"}.get(
        rec["localisation"], "pneumonia, no region")


@torch.no_grad()
def generate(model, tok, prompt, device, max_new):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, eos_token_id=tok.eos_token_id,
                             pad_token_id=tok.eos_token_id, repetition_penalty=1.05)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="LLM run folder from step 16")
    ap.add_argument("--n", type=int, default=300, help="test records to evaluate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = json.loads((args.run / "config.json").read_text())
    data = Path(cfg["data"])
    test_f = data / "llm_test.jsonl"
    if not test_f.exists():
        raise SystemExit("llm_test.jsonl missing: run 14_findings.py --split test, then 15_make_llm_data.py again.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.run / "adapter")
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model = PeftModel.from_pretrained(base, args.run / "adapter").to(device).eval()

    # one record per test X-ray, stratified so rare groups are represented
    recs = {}
    for e in map(json.loads, open(test_f)):
        recs.setdefault(e["id"], e["record"])
    by = collections.defaultdict(list)
    for rid, rec in recs.items():
        by[group_of(rec)].append((rid, rec))
    rng = random.Random(args.seed)
    per = max(1, args.n // len(by))
    chosen = [x for g in by.values() for x in rng.sample(g, min(per, len(g)))]
    out = args.run / "eval_test"; out.mkdir(exist_ok=True)

    rows, t0 = [], time.time()
    for i, (rid, rec) in enumerate(chosen, 1):
        rep = generate(model, tok, L.prompt(rec), device, 260)
        rows.append({"id": rid, "group": group_of(rec), "kind": "report", "question": None,
                     "text": rep, "problems": L.check(rec, rep, "report")})
        for q, ref, k in L.make_qa(rec, rng, 3):
            a = generate(model, tok, L.prompt(rec, q), device, 90)
            prob = L.check_answer(rec, k, a)        # no invented facts AND the asked-for fact present
            rows.append({"id": rid, "group": group_of(rec), "kind": k, "question": q, "text": a, "problems": prob})
        print(f"\r  {i}/{len(chosen)} records, {(time.time() - t0) / 60:.1f} min", end="")
    print()
    with open(out / "generations.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    lines = []
    def say(s=""):
        print(s); lines.append(s)
    say(f"LLM EVALUATION on TEST findings - {args.run.name} - model {cfg['model']}")
    say(f"records {len(chosen)} (" + ", ".join(f"{g} {len(v)}" for g, v in by.items()) + " available)")
    R = [r for r in rows if r["kind"] == "report"]; Q = [r for r in rows if r["kind"] != "report"]
    halluc = lambda r: any(not p.startswith("missing") for p in r["problems"])
    omit = lambda r: any(p.startswith("missing") for p in r["problems"])
    say(); say("=" * 80); say("1. REPORTS"); say("=" * 80)
    say(f"{'group':32s}{'n':>5s}{'faithful':>10s}{'hallucination':>15s}{'omission':>10s}")
    for g in sorted({r['group'] for r in R}) + ["ALL"]:
        d = R if g == "ALL" else [r for r in R if r["group"] == g]
        say(f"{g:32s}{len(d):5d}{sum(not r['problems'] for r in d) / len(d):10.3f}"
            f"{sum(halluc(r) for r in d) / len(d):15.3f}{sum(omit(r) for r in d) / len(d):10.3f}")
    c = collections.Counter(p.split(" [")[0] for r in R for p in r["problems"])
    say("problem types: " + (", ".join(f"{k} {v}" for k, v in c.most_common()) or "none"))
    say(); say("=" * 80); say("2. QUESTIONS"); say("=" * 80)
    say(f"{'question kind':20s}{'n':>5s}{'correct':>10s}")
    for k in sorted({r['kind'] for r in Q}) + ["ALL"]:
        d = Q if k == "ALL" else [r for r in Q if r["kind"] == k]
        say(f"{k:20s}{len(d):5d}{sum(not r['problems'] for r in d) / len(d):10.3f}")
    say(); say("Examples with problems (first 5):")
    for r in [r for r in rows if r["problems"]][:5]:
        say(f"- [{r['kind']}] {r['question'] or ''}\n  {r['text'][:400]}\n  -> {r['problems']}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved report.txt and generations.jsonl in {out}")


if __name__ == "__main__":
    main()
