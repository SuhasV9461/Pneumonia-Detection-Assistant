"""
Step 3 — Train the classifier: Normal (0) vs Lung Opacity (1)
==============================================================

Uses data_pipeline.py (step 2 of the old numbering: the loader, weights, augmentation, view AUROC)
and classifier_utils.py (the model). Writes everything into one run folder:

    runs/<backbone>_<img>_<date-time>/
        config.json           every setting of this run (later steps read it)
        best.keras            the weights with the best validation within-view AUROC
        history.csv           loss / AUROC per epoch, both stages
        val_predictions.csv   probability AND logit for every validation image (step 4 needs both)
        val_metrics.json      the validation numbers printed at the end

Training happens in two stages
  Stage A (warm-up):  backbone frozen, only the new head learns.        lr 1e-3, few epochs
  Stage B (fine-tune): last block(s) of the backbone also learn.         lr 2e-5, early stopping
  Why: the new head starts random. If the backbone were trainable from the start, the large,
  random gradients from the head would flow into the pretrained layers and damage them.

The TEST split is not touched here. It is used once, in the evaluation step, after the
threshold and calibration are fixed on validation.

Run (inside WSL, from the folder that holds data_pipeline.py and classifier_utils.py):

    # 1) 2-minute smoke test: tiny subset, 1 epoch per stage. Checks that everything runs.
    python 03_train_classifier.py --csv ~/rsna/outputs/metadata.csv --smoke

    # 2) the real run
    python 03_train_classifier.py --csv ~/rsna/outputs/metadata.csv

    # options: --backbone resnet50   --img-size 384 --batch-size 12   --unfreeze conv4,conv5
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# GPU memory growth (take memory as needed instead of grabbing it all at start-up) can only be
# switched on BEFORE anything has run on the GPU. So it is done here, first thing, before the
# other project files are imported. If TensorFlow has already started the GPU anyway, we just
# continue without it: for a single training run that is harmless.
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError as _e:
        print(f"Note: GPU memory growth not enabled ({_e}). Continuing - this is fine.")

from classifier_utils import BACKBONES, build_model, logit_model, preprocess_fn, set_trainable
from data_pipeline import (ViewAUCCallback, add_group_weights, evaluate_by_view, load_metadata,
                           make_dataset, print_metrics)

SEED = 42


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="metadata.csv from step 1")
    ap.add_argument("--runs", type=Path, default=None, help="where run folders go (default: <csv folder>/runs)")
    ap.add_argument("--backbone", choices=list(BACKBONES), default="densenet121")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs-head", type=int, default=4, help="stage A epochs")
    ap.add_argument("--epochs-ft", type=int, default=25, help="stage B max epochs (early stopping)")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=2e-5)
    ap.add_argument("--patience", type=int, default=5, help="early-stopping patience (epochs)")
    ap.add_argument("--unfreeze", default=None,
                    help="comma list of layer-name prefixes to unfreeze in stage B "
                         "(default conv5; 'all' = whole backbone)")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--include-not-normal", action="store_true",
                    help="ablation: Not-Normal images count as negatives (label 0)")
    ap.add_argument("--no-mixed-precision", action="store_true")
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or 'none' (none = testing only)")
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch per stage")
    ap.add_argument("--tag", default="", help="extra word in the run-folder name, e.g. clahe")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="random seed (weights init of the head, shuffling, augmentation). "
                         "The data split never changes - it comes from metadata.csv")
    ap.add_argument("--no-group-weights", action="store_true",
                    help="ablation A4: every training image gets weight 1 (no view balancing)")
    return ap.parse_args()


def setup_gpu(mixed):
    gpus = tf.config.list_physical_devices("GPU")   # (memory growth is set at the top of the file)
    if mixed and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"GPUs: {[g.name for g in gpus] or 'none (CPU only - this will be slow)'}"
          f" | mixed precision: {bool(mixed and gpus)}")


def subset_for_smoke(df, n_per_split=64):
    """Small subset with every (label, view) group present in every split, so the
    within-view AUROC can be computed (it needs both labels inside AP and inside PA)."""
    parts = []
    for split, g in df.groupby("split"):
        parts.append(g.sample(frac=1, random_state=SEED)
                      .groupby(["target", "view"]).head(n_per_split // 4))
    return pd.concat(parts).reset_index(drop=True)


class ViewAUCFallback(tf.keras.callbacks.Callback):
    """Safety net, placed right after ViewAUCCallback.
    The within-view AUROC only exists when BOTH labels appear inside AP and inside PA in the
    validation set. If it is missing, use the overall validation AUROC instead (with a warning),
    so checkpointing / early stopping still work instead of silently doing nothing."""

    def __init__(self):
        super().__init__()
        self.warned = False

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and "val_view_auc" not in logs:
            logs["val_view_auc"] = logs.get("val_auc", 0.0)
            if not self.warned:
                print("  !! within-view AUROC unavailable (one view has only one label in val) "
                      "-> using overall val AUROC to pick the best epoch")
                self.warned = True


def main():
    args = parse_args()
    setup_gpu(not args.no_mixed_precision)
    tf.keras.utils.set_random_seed(args.seed)

    # ------------------------------------------------------------------ data
    df = load_metadata(args.csv, include_not_normal=args.include_not_normal)
    if args.smoke:
        df = subset_for_smoke(df)
        args.epochs_head, args.epochs_ft = 1, 1
    df, group_w = add_group_weights(df)
    if args.no_group_weights:                      # ablation: switch the view balancing off
        df.loc[df["split"] == "train", "weight"] = 1.0
        group_w["weight"] = 1.0
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    prep = preprocess_fn(args.backbone)
    train_ds = make_dataset(df, "train", args.img_size, args.batch_size, training=True, seed=args.seed)
    train_ds = train_ds.map(lambda x, y, w: (prep(x), y, w), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = make_dataset(df, "val", args.img_size, args.batch_size, training=False)
    val_ds = val_ds.map(lambda x, y: (prep(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    print(f"\nTrain: {len(train_df)} images ({int(train_df['target'].sum())} opacity) | "
          f"Val: {len(val_df)} images ({int(val_df['target'].sum())} opacity)")
    print("Group weights (train):\n" + group_w.to_string(index=False))

    # ------------------------------------------------------------------ run folder
    runs = args.runs or args.csv.parent / "runs"
    tag = f"{args.backbone}_{args.img_size}" + ("_notnormal" if args.include_not_normal else "") \
          + ("_smoke" if args.smoke else "") + ("_noweights" if args.no_group_weights else "") \
          + (f"_seed{args.seed}" if args.seed != SEED else "") + (f"_{args.tag}" if args.tag else "")
    run_dir = runs / f"{tag}_{datetime.now():%Y%m%d-%H%M}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(seed=args.seed, run_dir=str(run_dir), n_train=len(train_df), n_val=len(val_df),
                  tf_version=tf.__version__)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Run folder: {run_dir}\n")

    # ------------------------------------------------------------------ model
    weights = None if args.weights.lower() == "none" else args.weights
    model, base = build_model(args.backbone, args.img_size, args.dropout, weights)
    auc = tf.keras.metrics.AUC(name="auc")
    best_path = run_dir / "best.keras"
    t0 = time.time()

    # ---------------- stage A: head only
    print("=" * 70 + "\nSTAGE A — backbone frozen, training the head\n" + "=" * 70)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr_head),
                  loss=tf.keras.losses.BinaryCrossentropy(), weighted_metrics=[auc])
    print(f"Trainable weights: {sum(int(np.prod(w.shape)) for w in model.trainable_weights):,}")
    hist_a = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head,
                       callbacks=[ViewAUCCallback(val_df, val_ds)], verbose=2)

    # ---------------- stage B: fine-tune
    print("=" * 70 + "\nSTAGE B — fine-tuning the last block(s) of the backbone\n" + "=" * 70)
    prefixes = args.unfreeze.split(",") if args.unfreeze else None
    n_tensors = set_trainable(base, args.backbone, prefixes)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr_ft),     # recompile after changing trainable
                  loss=tf.keras.losses.BinaryCrossentropy(), weighted_metrics=[auc])
    print(f"Unfrozen backbone tensors: {n_tensors} | trainable weights: "
          f"{sum(int(np.prod(w.shape)) for w in model.trainable_weights):,}")
    callbacks = [
        ViewAUCCallback(val_df, val_ds),            # must be FIRST: it adds val_view_auc to the logs
        ViewAUCFallback(),                          # fills val_view_auc if it could not be computed
        tf.keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_view_auc", mode="max",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_view_auc", mode="max",
                                         patience=args.patience, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_view_auc", mode="max", factor=0.3,
                                             patience=max(2, args.patience // 2), min_lr=1e-7, verbose=1),
    ]
    hist_b = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_ft,
                       callbacks=callbacks, verbose=2)
    minutes = (time.time() - t0) / 60

    # ------------------------------------------------------------------ history
    ha = pd.DataFrame(hist_a.history).assign(stage="A")
    hb = pd.DataFrame(hist_b.history).assign(stage="B")
    hist = pd.concat([ha, hb], ignore_index=True)
    hist.insert(0, "epoch", range(1, len(hist) + 1))
    hist.to_csv(run_dir / "history.csv", index=False)

    # ------------------------------------------------------------------ best model on validation
    if not best_path.exists():   # should not happen, but never lose a finished run
        print("  !! no checkpoint was written - saving the final weights as best.keras")
        model.save(str(best_path))
    model = tf.keras.models.load_model(str(best_path), compile=False)
    lm = logit_model(model)
    logits = lm.predict(val_ds, verbose=0).ravel().astype("float64")
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = val_df[["patientId", "class", "target", "view", "sex", "age"]].copy()
    preds["logit"] = logits
    preds["prob"] = probs
    preds.to_csv(run_dir / "val_predictions.csv", index=False)

    m = evaluate_by_view(val_df["target"].values, probs, val_df["view"].values, threshold=0.5)
    m = {k: (float(v) if not isinstance(v, int) else v) for k, v in m.items()}
    m.update(best_epoch_in_stage_B=int(np.argmax(hb["val_view_auc"].values)) + 1
             if "val_view_auc" in hb else None, train_minutes=round(minutes, 1))
    (run_dir / "val_metrics.json").write_text(json.dumps(m, indent=2))
    print_metrics(m, title="VALIDATION — best checkpoint (threshold 0.5; the real threshold is set in step 4)")
    print(f"\nTraining time: {minutes:.1f} min\nSaved everything in {run_dir}")


if __name__ == "__main__":
    main()
