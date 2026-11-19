"""
Step 2 — Input pipeline (tf.data) for RSNA Normal vs Opacity
=============================================================

This file is a *module*: later scripts (training, evaluation, Grad-CAM) import from it,
so every step loads data in exactly the same way.

    from data_pipeline import load_metadata, make_dataset, ViewAUCCallback, evaluate_by_view

What it does, and which audit finding each part answers:

  1. load_metadata()      keep Normal + Opacity, set impossible ages (>100) to missing,
                          add a "group" column (class x view)
  2. add_group_weights()  weight training images so normal-AP, normal-PA, opacity-AP and
                          opacity-PA count equally            -> breaks "AP = opacity"
  3. make_dataset()       tf.data loader: read PNG -> resize -> (augment if training) -> 3 channels
  4. augment()            geometry (small rotate / zoom / shift)
                          intensity (brightness, contrast, gamma, blur, noise) -> hides scanner "style"
                          corner erasing                                        -> hides R/L markers, text
  5. evaluate_by_view()   AUROC overall, within AP, within PA  (+ sensitivity / specificity)
  6. ViewAUCCallback      computes those on the validation set after every epoch, so the model
                          can be chosen by *within-view* AUROC instead of overall AUROC

Output images are float32 in the range 0-255 with 3 channels, which is what every
tf.keras.applications model's preprocess_input() expects. The model script (step 3) applies
the backbone-specific preprocess_input.

Run this file directly to check everything works and to SEE the augmentations:

    python data_pipeline.py --csv "/home/suhas/AIML Project/Test 2/RSNA/outputs/metadata.csv"

It prints the dataset sizes and weights, times the loader, and saves
augmentation_preview.png next to metadata.csv — open it and check the X-rays still look
like X-rays.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

AUTOTUNE = tf.data.AUTOTUNE

# ----------------------------------------------------------------------------
# Augmentation settings — change here, everything else follows
# ----------------------------------------------------------------------------
AUG = dict(
    rotate_deg=10,          # random rotation in [-10, +10] degrees
    zoom=0.10,              # random scale in [0.90, 1.10]
    shift=0.06,             # random shift up to 6 % of the image size
    hflip=False,            # left-right flip: off (puts the heart on the wrong side)
    brightness=0.10,        # add a random value in [-0.10, +0.10]
    contrast=(0.80, 1.20),  # stretch / squeeze contrast around the mean
    gamma=(0.80, 1.25),     # non-linear brightness change (different scanners' curves)
    p_blur=0.20,            # probability of a slight Gaussian blur
    p_noise=0.20,           # probability of adding grain
    noise_std=0.02,
    p_erase=0.50,           # probability of erasing patches near the corners
    erase_max_patches=2,
    erase_size=(0.08, 0.18),  # patch side, as a fraction of the image side
    erase_corner_zone=0.25,   # patches are placed inside the outer 25 % (where markers live)
)


# ----------------------------------------------------------------------------
# 1. Metadata
# ----------------------------------------------------------------------------
def load_metadata(csv_path, include_not_normal=False):
    """Read metadata.csv from step 1 and prepare it for training.

    include_not_normal=False -> Normal (0) vs Opacity (1)           [current task]
    include_not_normal=True  -> Normal + Not-Normal (0) vs Opacity (1)  [hard-negative experiment]
    """
    df = pd.read_csv(csv_path)
    keep = ["normal", "opacity"] + (["not_normal"] if include_not_normal else [])
    df = df[df["class"].isin(keep)].reset_index(drop=True)

    # 5 patients have ages 148-155 (data-entry errors). Age is not a model input, but
    # the per-age-group results need clean values, so mark them as missing.
    df.loc[(df["age"] > 100) | (df["age"] < 0), "age"] = np.nan

    df["target"] = (df["class"] == "opacity").astype(int)
    df["group"] = df["class"] + "_" + df["view"]
    df["weight"] = 1.0
    return df


# ----------------------------------------------------------------------------
# 2. Group weights (training split only)
# ----------------------------------------------------------------------------
def add_group_weights(df):
    """Give every (label, view) group the same total weight in the TRAINING split.

    weight = N_train / (number_of_groups x N_group)
    so a rare group (e.g. opacity-PA) gets a big weight and a common one a small weight.
    Validation and test keep weight 1 -> they are measured on the natural mix.
    """
    df = df.copy()
    tr = df["split"] == "train"
    counts = df[tr].groupby(["target", "view"]).size()
    n_groups = len(counts)
    w = len(df[tr]) / (n_groups * counts)
    df.loc[tr, "weight"] = [w[(t, v)] for t, v in zip(df.loc[tr, "target"], df.loc[tr, "view"])]
    return df, w.rename("weight").reset_index()


# ----------------------------------------------------------------------------
# 3. Loading one image
# ----------------------------------------------------------------------------
def _load_png(path, img_size):
    """PNG file -> uint8 tensor (img_size, img_size, 1)."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_png(raw, channels=1)
    img = tf.image.resize(img, (img_size, img_size), method="area")
    return tf.cast(tf.round(img), tf.uint8)


# ----------------------------------------------------------------------------
# 4. Augmentation (training only). Works on float images in [0, 1], shape (H, W, 1)
# ----------------------------------------------------------------------------
def _random_affine(img, cfg):
    """Small random rotation + zoom + shift, done in one transform."""
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    angle = tf.random.uniform([], -cfg["rotate_deg"], cfg["rotate_deg"]) * math.pi / 180.0
    scale = tf.random.uniform([], 1.0 - cfg["zoom"], 1.0 + cfg["zoom"])
    tx = tf.random.uniform([], -cfg["shift"], cfg["shift"]) * w
    ty = tf.random.uniform([], -cfg["shift"], cfg["shift"]) * h

    # The transform maps OUTPUT pixel (x, y) -> INPUT pixel, around the image centre.
    cos, sin = tf.cos(angle) / scale, tf.sin(angle) / scale
    cx, cy = w / 2.0, h / 2.0
    a0, a1 = cos, -sin
    b0, b1 = sin, cos
    a2 = cx - a0 * cx - a1 * cy - tx
    b2 = cy - b0 * cx - b1 * cy - ty
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]

    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None], transforms=transform,
        output_shape=tf.shape(img)[:2], fill_value=0.0,
        interpolation="BILINEAR", fill_mode="CONSTANT")
    return out[0]


def _gaussian_blur(img, sigma):
    """3x3 Gaussian blur."""
    x = tf.constant([-1.0, 0.0, 1.0])
    k1 = tf.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k1 = k1 / tf.reduce_sum(k1)
    k = tf.tensordot(k1, k1, axes=0)[:, :, None, None]
    return tf.nn.depthwise_conv2d(img[None], k, [1, 1, 1, 1], "SAME")[0]


def _erase_corners(img, cfg):
    """Paint 1..N random flat patches inside the outer border zone (where R/L markers and
    text usually are). The model can never rely on what is in the corners."""
    size = tf.shape(img)[0]
    s = tf.cast(size, tf.float32)
    n = tf.random.uniform([], 1, cfg["erase_max_patches"] + 1, dtype=tf.int32)
    zone = cfg["erase_corner_zone"]

    def one_patch(i, im):
        side = tf.cast(tf.random.uniform([], *cfg["erase_size"]) * s, tf.int32)
        # pick a corner: top/bottom and left/right
        top = tf.random.uniform([]) < 0.5
        left = tf.random.uniform([]) < 0.5
        max_off = tf.maximum(tf.cast(zone * s, tf.int32) - side, 1)
        oy = tf.random.uniform([], 0, max_off, dtype=tf.int32)
        ox = tf.random.uniform([], 0, max_off, dtype=tf.int32)
        y0 = tf.where(top, oy, size - side - oy)
        x0 = tf.where(left, ox, size - side - ox)
        yy = tf.range(size)[:, None]
        xx = tf.range(size)[None, :]
        mask = (yy >= y0) & (yy < y0 + side) & (xx >= x0) & (xx < x0 + side)
        fill = tf.random.uniform([])                    # random grey level
        im = tf.where(mask[:, :, None], fill, im)
        return i + 1, im

    _, img = tf.while_loop(lambda i, im: i < n, one_patch, [0, img])
    return img


def augment(img, cfg=AUG):
    """img: float32 (H, W, 1) in [0, 1] -> augmented image, same shape and range."""
    # geometry
    img = _random_affine(img, cfg)
    if cfg["hflip"]:
        img = tf.image.random_flip_left_right(img)

    # intensity -> hides each scanner's brightness/contrast "style"
    img = img + tf.random.uniform([], -cfg["brightness"], cfg["brightness"])
    mean = tf.reduce_mean(img)
    img = (img - mean) * tf.random.uniform([], *cfg["contrast"]) + mean
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.pow(img, tf.random.uniform([], *cfg["gamma"]))

    if tf.random.uniform([]) < cfg["p_blur"]:
        img = _gaussian_blur(img, tf.random.uniform([], 0.5, 1.2))
    if tf.random.uniform([]) < cfg["p_noise"]:
        img = img + tf.random.normal(tf.shape(img), stddev=cfg["noise_std"])

    # markers / text
    if tf.random.uniform([]) < cfg["p_erase"]:
        img = _erase_corners(img, cfg)

    return tf.clip_by_value(img, 0.0, 1.0)


# ----------------------------------------------------------------------------
# 3. The dataset
# ----------------------------------------------------------------------------
def make_dataset(df, split, img_size=224, batch_size=32, training=None,
                 cache=None, seed=42, with_weights=None):
    """Build a tf.data.Dataset for one split.

    training=True  -> shuffle + augment, yields (image, label, weight)   [use for model.fit]
    training=False -> no shuffle, no augment, yields (image, label)      [val / test / predict]
    cache          -> keep decoded images in RAM after the first epoch (much faster).
                      Default: on when img_size <= 256 (~0.5 GB for the training set).
    """
    part = df[df["split"] == split].reset_index(drop=True)
    if training is None:
        training = split == "train"
    if with_weights is None:
        with_weights = training
    if cache is None:
        cache = img_size <= 256

    ds = tf.data.Dataset.from_tensor_slices((
        part["png_path"].values,
        part["target"].values.astype("float32"),
        part["weight"].values.astype("float32"),
    ))
    ds = ds.map(lambda p, y, w: (_load_png(p, img_size), y, w), num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()                       # cached BEFORE augmentation -> new augmentations every epoch
    if training:
        ds = ds.shuffle(len(part), seed=seed, reshuffle_each_iteration=True)

    def finish(img, y, w):
        img = tf.cast(img, tf.float32) / 255.0
        if training:
            img = augment(img)
        img = tf.image.grayscale_to_rgb(img) * 255.0      # 3 channels, 0-255 for preprocess_input
        img = tf.ensure_shape(img, (img_size, img_size, 3))
        return (img, y, w) if with_weights else (img, y)

    ds = ds.map(finish, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=training)
    return ds.prefetch(AUTOTUNE)


# ----------------------------------------------------------------------------
# 5. Evaluation: overall + within each view
# ----------------------------------------------------------------------------
def evaluate_by_view(y_true, y_prob, views, threshold=0.5):
    """Returns a dict of metrics. 'view_auc' = mean of AP and PA AUROC = the number we
    choose models by, because inside one view the view itself cannot help."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).ravel()
    views = np.asarray(views)
    out = {}

    def block(name, m):
        yt, yp = y_true[m], y_prob[m]
        if len(np.unique(yt)) < 2:
            return
        pred = yp >= threshold
        tp = np.sum(pred & (yt == 1)); fn = np.sum(~pred & (yt == 1))
        tn = np.sum(~pred & (yt == 0)); fp = np.sum(pred & (yt == 0))
        out[f"auc{name}"] = roc_auc_score(yt, yp)
        out[f"sens{name}"] = tp / max(tp + fn, 1)
        out[f"spec{name}"] = tn / max(tn + fp, 1)
        out[f"n{name}"] = int(m.sum())

    block("", np.ones(len(y_true), bool))
    for v in ("AP", "PA"):
        block(f"_{v}", views == v)
    if "auc_AP" in out and "auc_PA" in out:
        out["view_auc"] = (out["auc_AP"] + out["auc_PA"]) / 2
    return out


def print_metrics(m, title="Results"):
    print(f"\n{title}")
    print(f"  {'':8s} {'AUROC':>7s} {'Sens':>7s} {'Spec':>7s} {'n':>6s}")
    for key, label in (("", "overall"), ("_AP", "AP"), ("_PA", "PA")):
        if f"auc{key}" in m:
            print(f"  {label:8s} {m['auc'+key]:7.3f} {m['sens'+key]:7.3f} "
                  f"{m['spec'+key]:7.3f} {m['n'+key]:6d}")
    if "view_auc" in m:
        print(f"  within-view AUROC (mean of AP, PA): {m['view_auc']:.3f}")


# ----------------------------------------------------------------------------
# 6. Keras callback: within-view AUROC on validation after every epoch
# ----------------------------------------------------------------------------
class ViewAUCCallback(tf.keras.callbacks.Callback):
    """Adds val_auc, val_auc_AP, val_auc_PA and val_view_auc to the training logs.

    IMPORTANT: put this callback BEFORE ModelCheckpoint / EarlyStopping in the callbacks
    list, so they can use monitor="val_view_auc", mode="max".
    """

    def __init__(self, val_df, val_ds):
        super().__init__()
        self.y = val_df["target"].values
        self.views = val_df["view"].values
        self.ds = val_ds

    def on_epoch_end(self, epoch, logs=None):
        prob = self.model.predict(self.ds, verbose=0).ravel()
        m = evaluate_by_view(self.y, prob, self.views)
        if logs is not None:
            for k in ("auc", "auc_AP", "auc_PA", "view_auc"):
                if k in m:
                    logs[f"val_{k}"] = m[k]
        print(f"  -> val AUROC {m.get('auc', float('nan')):.3f} | AP {m.get('auc_AP', float('nan')):.3f}"
              f" | PA {m.get('auc_PA', float('nan')):.3f} | within-view {m.get('view_auc', float('nan')):.3f}")


# ----------------------------------------------------------------------------
# Self-check: run this file directly
# ----------------------------------------------------------------------------
def _preview(df, out_png, img_size, n=6, seed=0):
    """Save a grid: each row = one training image, column 1 = original, then 5 augmentations."""
    import cv2
    rng = np.random.default_rng(seed)
    tr = df[df["split"] == "train"]
    rows = []
    for idx in rng.choice(len(tr), size=min(n, len(tr)), replace=False):
        r = tr.iloc[idx]
        base = tf.cast(_load_png(r["png_path"], img_size), tf.float32) / 255.0
        tiles = [base.numpy()] + [augment(base).numpy() for _ in range(5)]
        tiles = [(t[:, :, 0] * 255).astype(np.uint8) for t in tiles]
        label = f"{r['class']} {r['view']}"
        cv2.putText(tiles[0], label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(out_png), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path.home() / "AIML Project" / "Test 2" / "RSNA" / "outputs" / "metadata.csv")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    df = load_metadata(args.csv)
    df, weights = add_group_weights(df)

    print("Images per split (Normal / Opacity):")
    print(pd.crosstab(df["split"], df["class"]).to_string(), "\n")
    print("Training group weights (each group now counts equally):")
    print(weights.to_string(index=False), "\n")
    tr = df[df["split"] == "train"]
    eff = tr.groupby(["target", "view"])["weight"].sum().unstack()
    print("Effective (weighted) training counts — should all be equal:")
    print(eff.round(0).to_string(), "\n")

    ds = make_dataset(df, "train", args.img_size, args.batch_size)
    x, y, w = next(iter(ds))
    print(f"One training batch: images {tuple(x.shape)} {x.dtype}, range "
          f"{float(tf.reduce_min(x)):.0f}-{float(tf.reduce_max(x)):.0f}; labels {tuple(y.shape)}; "
          f"weights {tuple(w.shape)}")

    t0, n = time.time(), 0
    for i, (x, y, w) in enumerate(ds):
        n += x.shape[0]
        if i == 30:
            break
    print(f"Loader speed (first pass, no cache yet): {n / (time.time() - t0):.0f} images/s")

    out_png = args.csv.parent / "augmentation_preview.png"
    _preview(df, out_png, args.img_size)
    print(f"Saved {out_png} — column 1 is the original, the other 5 are augmentations.")


if __name__ == "__main__":
    main()
