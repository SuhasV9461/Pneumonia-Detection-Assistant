"""
Step 16 — Fine-tune GPT-Neo with LoRA on the findings -> text data (Phase 3)
===========================================================================

Run in yolo-env (PyTorch) after:  pip install transformers peft

What LoRA is
  GPT-Neo 1.3B has 1.3 billion weights. Training all of them needs far more than 8 GB.
  LoRA freezes every original weight and adds, next to some weight matrices W, two small
  matrices A (d x r) and B (r x d) with r = 16. The layer then computes W x + B A x.
  Only A and B are trained: about 0.5 % of the weights. A starts random and B starts at
  zero, so at step 0 the model is exactly the original GPT-Neo.
  The trained "adapter" (A and B) is a small file, saved separately from GPT-Neo.

What is trained
  Each example = prompt (findings + instruction) + response + end-of-text token.
  The loss is computed ONLY on the response tokens (prompt tokens get label -100, which
  the loss ignores). The model learns to WRITE the response, not to repeat the findings.

Settings
  model  EleutherAI/gpt-neo-1.3B in bfloat16 (2.6 GB on the GPU); --model EleutherAI/gpt-neo-125M
         is the fast fallback
  LoRA   r 16, alpha 32, dropout 0.05, on the attention (q, k, v, out) and MLP (c_fc, c_proj)
  batch  8 x 2 accumulation = 16 examples per update; learning rate 2e-4 with 50 warm-up
         steps then linear decay; 1 epoch (~1,200 updates for ~19,000 examples)
  memory gradient checkpointing (recompute activations instead of storing them)
  check  validation loss on 400 llm_val examples every 200 updates; the best adapter is kept
  end    writes 6 reports for llm_val records with greedy decoding and runs the
         faithfulness check from llm_utils on them

Output: <outputs>/llm_runs/<name>/  adapter/ (LoRA weights + tokenizer), log.csv, config.json,
        samples.txt
Run (inside WSL, yolo-env active, from the code folder):
    pip install transformers peft
    python 16_train_llm.py --data "$HOME/AIML Project/Test 2/RSNA/outputs/llm_data" --smoke   # 125M, 100 updates
    python 16_train_llm.py --data "$HOME/AIML Project/Test 2/RSNA/outputs/llm_data"
"""

import argparse
import csv
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import llm_utils as L


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def encode(tok, ex, max_len):
    p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
    r = tok(ex["response"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (p + r)[:max_len]
    labels = ([-100] * len(p) + r)[:max_len]          # learn the response only
    return ids, labels


def batches(data, bs, pad_id, shuffle, rng):
    idx = list(range(len(data)))
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, len(idx), bs):
        chunk = [data[i] for i in idx[s:s + bs]]
        n = max(len(c[0]) for c in chunk)
        ids = torch.full((len(chunk), n), pad_id); lab = torch.full((len(chunk), n), -100)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for j, (a, b) in enumerate(chunk):
            ids[j, :len(a)] = torch.tensor(a); lab[j, :len(b)] = torch.tensor(b); att[j, :len(a)] = 1
        yield ids, lab, att


@torch.no_grad()
def val_loss(model, data, bs, pad_id, device):
    model.eval(); tot = n = 0
    for ids, lab, att in batches(data, bs, pad_id, False, None):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device))
        k = int((lab[:, 1:] != -100).sum()); tot += out.loss.item() * k; n += k
    model.train()
    return tot / max(n, 1)


@torch.no_grad()
def generate(model, tok, prompt, device, max_new=220):
    # writing text is not training: evaluation mode, and no gradient checkpointing
    # (checkpointing is a training-only memory trick; it was left on here before - fixed)
    model.eval()
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, eos_token_id=tok.eos_token_id,
                             pad_token_id=tok.eos_token_id, repetition_penalty=1.05)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="llm_data folder from step 15")
    ap.add_argument("--model", default="EleutherAI/gpt-neo-1.3B")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="gpt-neo-125M, 100 updates")
    args = ap.parse_args()
    if args.smoke and args.model == "EleutherAI/gpt-neo-1.3B":
        args.model = "EleutherAI/gpt-neo-125M"
    torch.manual_seed(args.seed); rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    name = Path(args.model).name + ("_smoke" if args.smoke else "") + datetime.now().strftime("_%Y%m%d-%H%M")
    out = args.data.parent / "llm_runs" / name
    (out / "adapter").mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()                         # needed for checkpointing + LoRA
    lcfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj"])
    model = get_peft_model(model, lcfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {total / 1e6:.0f} M weights, LoRA trains {trainable / 1e6:.2f} M ({100 * trainable / total:.2f} %)")

    train = [encode(tok, e, args.max_len) for e in load_jsonl(args.data / "llm_train.jsonl")]
    val_ex = load_jsonl(args.data / "llm_val.jsonl")
    val = [encode(tok, e, args.max_len) for e in random.Random(0).sample(val_ex, min(400, len(val_ex)))]
    steps = 100 if args.smoke else math.ceil(len(train) / (args.batch * args.accum) * args.epochs)
    print(f"training examples {len(train)}, updates {steps}, device {device}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps))
    log = open(out / "log.csv", "w", newline=""); w = csv.writer(log); w.writerow(["update", "train_loss", "val_loss", "minutes"])
    best, step, micro, run_loss, t0 = float("inf"), 0, 0, [], time.time()
    model.train()
    while step < steps:
        for ids, lab, att in batches(train, args.batch, tok.pad_token_id, True, rng):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device)).loss
            (loss / args.accum).backward(); run_loss.append(loss.item()); micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 20 == 0:
                print(f"\r update {step}/{steps}  train loss {sum(run_loss[-40:]) / len(run_loss[-40:]):.4f}  "
                      f"{(time.time() - t0) / 60:.1f} min", end="")
            if step % args.eval_every == 0 or step == steps:
                vl = val_loss(model, val, args.batch, tok.pad_token_id, device)
                w.writerow([step, round(sum(run_loss[-40:]) / len(run_loss[-40:]), 4), round(vl, 4),
                            round((time.time() - t0) / 60, 1)]); log.flush()
                print(f"\n update {step}: validation loss {vl:.4f}" + ("  (best, saved)" if vl < best else ""))
                if vl < best:
                    best = vl; model.save_pretrained(out / "adapter"); tok.save_pretrained(out / "adapter")
            if step >= steps:
                break
    log.close()

    # a first look: 6 reports on llm_val records, checked for faithfulness
    lines, n_bad = [], 0
    reports = [e for e in val_ex if e["kind"] == "report"]
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    print(f"\nBest validation loss {best:.4f}; adapter saved in {out / 'adapter'}. Writing 6 sample reports ...")
    for e in random.Random(1).sample(reports, min(6, len(reports))):
        try:
            gen = generate(model, tok, e["prompt"], device)
        except RuntimeError as err:      # training is already saved; do not lose it over a sample
            print(f"Could not write samples ({str(err)[:120]}). Training is saved; fix this before step 17.")
            return
        problems = L.check(e["record"], gen)
        n_bad += bool(problems)
        lines += ["=" * 80, e["prompt"] + gen, "-- faithfulness check: " + ("OK" if not problems else "; ".join(problems))]
    (out / "samples.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\n{n_bad} of {min(6, len(reports))} sample reports failed the faithfulness check (step 17 measures this properly).")
    print(f"Best validation loss {best:.4f}. Adapter, log and samples saved in {out}")


if __name__ == "__main__":
    main()
