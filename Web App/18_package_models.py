"""
Step 18 — Package every frozen model into ONE folder for the web application (Phase 4)
=====================================================================================

Run in ai-env (TensorFlow) after:  pip install tf2onnx onnx onnxruntime

Why ONNX
  The classifier is TensorFlow, the detector and GPT-Neo are PyTorch. Rather than run two
  environments behind the website, the classifier is converted ONCE to ONNX, an open model
  format that the small library onnxruntime can run anywhere (no TensorFlow needed). Then
  the whole website runs in yolo-env.

What it does
  1. For each member of the final classifier (final_model/final_config.json, step 10):
       - rebuilds the network in float32 and copies the trained weights into it (training
         used mixed precision; a float32 copy converts more cleanly and is what onnxruntime
         runs on a CPU anyway)
       - exports a model with TWO outputs:  the logit, and the last feature maps
         (12 x 12 x 1024 at 384 px). With global average pooling + one Dense layer,
         Grad-CAM equals  ReLU( sum_k w_k * featuremap_k ), so the heat-map needs no
         gradients: the Dense weights w are saved next to the model.
  2. CHECKS the export on 64 validation images:
       ONNX vs the TensorFlow model it came from      : must differ by < 0.05 (the conversion)
       ONNX vs the logits step 10 cached (cache_val)  : must differ by < 0.25. Step 10 ran in mixed
         precision (float16 inside the network), the export in float32, so small rounding
         differences are expected; 0.07 in logit is about 1 percentage point in probability.
       Whether the DECISIONS stay the same is checked by the self-test in webapp/pipeline.py.
  3. Copies everything the website needs into <outputs>/webapp_models/:
       classifier_<k>.onnx, cam_<k>.npz, calibration_v2.json, final_config.json,
       detector.pt, detector_config.json, findings_config.json,
       llm_adapter/ (LoRA weights + tokenizer) and manifest.json (what is where).

Run (inside WSL, ai-env active, from the code folder):
    pip install tf2onnx onnx onnxruntime
    python 18_package_models.py --outputs "$HOME/AIML Project/Test 2/RSNA/outputs" \\
        --detector-run "$HOME/AIML Project/Test 2/RSNA/outputs/detector_runs/yolo11s_512_20261028-0921" \\
        --llm-run "$HOME/AIML Project/Test 2/RSNA/outputs/llm_runs/gpt-neo-1.3B_20261101-0754"
"""

import tensorflow as tf

for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import tf2onnx

from classifier_utils import build_model, preprocess_fn
from data_pipeline import load_metadata, make_dataset

TOL_CONVERT = 0.05   # ONNX vs the TensorFlow model it was made from: must be (almost) identical
TOL_PRECISION = 0.25 # vs step 10's cache: training / step 10 ran in mixed precision (float16), the
                     # export runs in float32 -> rounding inside 121 layers differs slightly


def float32_twin(run_dir):
    cfg = json.loads((run_dir / "config.json").read_text())
    trained = tf.keras.models.load_model(str(run_dir / "best.keras"), compile=False)
    tf.keras.mixed_precision.set_global_policy("float32")
    twin, _ = build_model(cfg["backbone"], cfg["img_size"], cfg.get("dropout", 0.3), weights=None)
    twin.set_weights(trained.get_weights())                  # same architecture -> same order
    two = tf.keras.Model(twin.input, [twin.get_layer("logit").output, twin.get_layer("gap").input],
                         name="classifier_logit_and_features")
    w, b = twin.get_layer("logit").get_weights()
    return cfg, two, w[:, 0].astype("float32"), float(b[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, required=True, help="the RSNA outputs folder (has metadata.csv)")
    ap.add_argument("--detector-run", type=Path, required=True)
    ap.add_argument("--llm-run", type=Path, required=True)
    ap.add_argument("--n-check", type=int, default=64)
    args = ap.parse_args()
    fm = args.outputs / "final_model"
    out = args.outputs / "webapp_models"
    out.mkdir(exist_ok=True)
    final_cfg = json.loads((fm / "final_config.json").read_text())
    cache = np.load(fm / "cache_val.npz", allow_pickle=True)

    df = load_metadata(args.outputs / "metadata.csv", include_not_normal=False)
    val = df[df["split"] == "val"].reset_index(drop=True)
    chk = val.iloc[: args.n_check].copy()
    chk["split"] = "chk"
    df_chk = pd.concat([df, chk], ignore_index=True)
    cache_ids = list(cache["ids"])

    members = []
    for k, run in enumerate(Path(m) for m in final_cfg["members"]):
        print(f"[{k + 1}/{len(final_cfg['members'])}] {run.name}")
        cfg, two, w, b = float32_twin(run)
        size = cfg["img_size"]
        onnx_path = out / f"classifier_{k}.onnx"
        spec = (tf.TensorSpec((None, size, size, 3), tf.float32, name="image"),)
        tf2onnx.convert.from_keras(two, input_signature=spec, opset=17, output_path=str(onnx_path))
        np.savez(out / f"cam_{k}.npz", w=w, b=b)

        # ---- check: TensorFlow vs ONNX vs the logits cached by step 10
        prep = preprocess_fn(cfg["backbone"])
        x = np.concatenate([prep(xb).numpy() for xb, _ in
                            make_dataset(df_chk, "chk", size, 16, training=False, cache=False)]).astype("float32")
        z_tf = two.predict(x, verbose=0)[0].ravel()
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        z_onnx = sess.run(None, {sess.get_inputs()[0].name: x})[0].ravel()
        z_cache = cache[run.name][[cache_ids.index(p) for p in chk["patientId"]], 0]
        d1, d2 = float(np.abs(z_tf - z_onnx).max()), float(np.abs(z_onnx - z_cache).max())
        print(f"    max logit difference: ONNX vs TensorFlow {d1:.4f} (limit {TOL_CONVERT}) | "
              f"ONNX vs step-10 cache {d2:.4f} (limit {TOL_PRECISION}, float32 vs mixed precision)")
        if d1 > TOL_CONVERT or d2 > TOL_PRECISION:
            raise SystemExit("Export check FAILED - send me this output.")
        members.append({"onnx": onnx_path.name, "cam": f"cam_{k}.npz", "img_size": size,
                        "backbone": cfg["backbone"], "source_run": run.name})

    # ---- copy everything else
    for src, dst in ((fm / "calibration_v2.json", "calibration_v2.json"), (fm / "final_config.json", "final_config.json"),
                     (args.detector_run / "weights" / "best.pt", "detector.pt"),
                     (args.detector_run / "detector_config.json", "detector_config.json"),
                     (args.detector_run / "findings_config.json", "findings_config.json")):
        shutil.copy2(src, out / dst)
    if (out / "llm_adapter").exists():
        shutil.rmtree(out / "llm_adapter")
    shutil.copytree(args.llm_run / "adapter", out / "llm_adapter")
    llm_cfg = json.loads((args.llm_run / "config.json").read_text())

    manifest = {"classifier": {"members": members, "tta": final_cfg["tta"],
                               "normalisation": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
                "png_size": 512, "detector": {"weights": "detector.pt", "imgsz": 512},
                "llm": {"base_model": llm_cfg["model"], "adapter": "llm_adapter"},
                "files": sorted(p.name for p in out.iterdir())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nEverything the website needs is in {out}:")
    for p in sorted(out.iterdir()):
        size = sum(f.stat().st_size for f in p.rglob("*")) if p.is_dir() else p.stat().st_size
        print(f"  {p.name:28s} {size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
