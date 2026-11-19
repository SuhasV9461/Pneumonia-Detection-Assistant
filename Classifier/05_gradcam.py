"""
Step 5 — Grad-CAM: where does the classifier look?
===================================================

Input : a run folder from step 3 (best.keras, config.json) + calibration_v2.json from step 4b,
        metadata.csv (step 1) and boxes.csv (step 2).
Output: <run>/gradcam/
    gradcam_<split>.csv          one row per image: probability, heat-map peak, the scores below
    gradcam_summary.txt          the tables printed below, for your report
    gallery_confident_misses.png      pneumonia the model was surest was normal
    gallery_confident_false_alarms.png normal images the model was surest were pneumonia
    gallery_correct_pneumonia.png     random correctly found pneumonia
    gallery_correct_normal.png        random correctly cleared normals

How Grad-CAM works (for one image)
  1. Run the image through DenseNet up to its last convolutional layer: a 7 x 7 grid of
     1,024 feature maps. Each grid cell "sees" roughly a 32 x 32 pixel patch of the 224 image.
  2. Ask: if feature map k got a little stronger, how much would the pneumonia LOGIT rise?
     That is the gradient. Average it over the 7 x 7 grid -> one importance weight per map.
  3. Heat-map = weighted sum of the 1,024 maps, negative values set to zero (we only want
     regions that push TOWARDS pneumonia), then enlarged from 7 x 7 to the image size.

The scores (computed on every image of the split)
  pointing game   pneumonia images only: is the hottest pixel inside a radiologist box?
                  Chance level = how much of the image the boxes cover (printed next to it).
  energy in box   pneumonia images only: share of all heat-map "mass" that lies inside boxes.
  corner energy   all images: share of heat-map mass in the four corner squares (20 % x 20 %
                  each, 16 % of the image together). Markers, text and labels live there;
                  the lungs almost never do. Much more than 16 % would point to a shortcut.
  peak in corner  share of images whose hottest pixel is in a corner square.

Run (inside WSL; needs the GPU only for speed, about 1-2 minutes for validation):
    python 05_gradcam.py --run "$HOME/AIML Project/Test 2/RSNA/outputs/runs/densenet121_224_20261025-1247" \
                         --csv "$HOME/AIML Project/Test 2/RSNA/outputs/metadata.csv"
"""

import tensorflow as tf

for _gpu in tf.config.list_physical_devices("GPU"):        # same reason as in step 3
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import calibration_utils as cu
from classifier_utils import preprocess_fn
from data_pipeline import load_metadata, make_dataset

ORIG = 1024          # RSNA original image size (box coordinates are in this space)
CORNER = 0.20        # corner square side, as a fraction of the image side
SEED = 42


# ---------------------------------------------------------------------------- Grad-CAM
class GradCAM:
    """Batched Grad-CAM on the pre-sigmoid logit, using the model's own layers."""

    def __init__(self, model):
        subs = [l for l in model.layers if isinstance(l, tf.keras.Model)]
        if len(subs) != 1:
            raise ValueError("expected exactly one backbone sub-model inside the classifier")
        self.base = subs[0]                      # DenseNet121 / ResNet50 without its top
        self.gap = model.get_layer("gap")
        self.dense = model.get_layer("logit")
        # compiled (graph-mode) backbone: faster, and no gradient bookkeeping
        self.features = tf.function(lambda x: self.base(x, training=False))

    def __call__(self, x):
        """x: preprocessed batch (B, H, W, 3) -> (cams (B, h, w) float32, logits (B,))"""
        # The backbone runs OUTSIDE the GradientTape. We only need the gradient of the logit
        # with respect to the last feature maps, not through the whole network. Inside the
        # tape, TensorFlow would keep every intermediate DenseNet activation in GPU memory
        # (for possible gradients w.r.t. the unfrozen weights) -> out of memory.
        feats = self.features(x)                             # (B, 7, 7, 1024): last conv activations
        with tf.GradientTape() as tape:
            tape.watch(feats)
            logit = self.dense(self.gap(feats))             # (B, 1); dropout is off at inference
        grads = tape.gradient(logit, feats)                  # images are independent -> per-image grads
        feats = tf.cast(feats, tf.float32)
        grads = tf.cast(grads, tf.float32)
        w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)          # (B, 1, 1, 1024)
        cam = tf.nn.relu(tf.reduce_sum(w * feats, axis=-1))            # (B, 7, 7)
        return cam.numpy(), tf.cast(logit, tf.float32).numpy().ravel()


def upsample(cam, size):
    """7 x 7 -> size x size, bilinear, then scaled so the maximum is 1 (all-zero stays zero)."""
    big = cv2.resize(cam, (size, size), interpolation=cv2.INTER_LINEAR)
    m = big.max()
    return big / m if m > 0 else big


# ---------------------------------------------------------------------------- scores
def box_mask(rows, size):
    """Union of all boxes of one patient as a size x size boolean mask."""
    m = np.zeros((size, size), bool)
    s = size / ORIG
    for _, r in rows.iterrows():
        x0, y0 = int(np.floor(r.x * s)), int(np.floor(r.y * s))
        x1, y1 = int(np.ceil((r.x + r.w) * s)), int(np.ceil((r.y + r.h) * s))
        m[max(y0, 0):min(y1, size), max(x0, 0):min(x1, size)] = True
    return m


def corner_mask(size, frac=CORNER):
    c = int(round(frac * size))
    m = np.zeros((size, size), bool)
    m[:c, :c] = m[:c, -c:] = m[-c:, :c] = m[-c:, -c:] = True
    return m


# ---------------------------------------------------------------------------- galleries
def gallery(rows, cams, boxes, png_size, title, path, ncol=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if len(rows) == 0:
        return
    n = len(rows)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol * 2, figsize=(ncol * 2 * 2.6, nrow * 2.9), dpi=110)
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (_, r) in enumerate(rows.iterrows()):
        img = cv2.imread(r["png_path"], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (png_size, png_size), interpolation=cv2.INTER_AREA)
        heat = upsample(cams[r["_idx"]], png_size)
        a1, a2 = axes[i // ncol, (i % ncol) * 2], axes[i // ncol, (i % ncol) * 2 + 1]
        a1.imshow(img, cmap="gray")
        a2.imshow(img, cmap="gray")
        a2.imshow(heat, cmap="jet", alpha=0.40, vmin=0, vmax=1)
        s = png_size / ORIG
        for _, b in boxes[boxes["patientId"] == r["patientId"]].iterrows():
            for a in (a1, a2):
                a.add_patch(Rectangle((b.x * s, b.y * s), b.w * s, b.h * s,
                                      fill=False, ec="#33ff66", lw=1.4))
        truth = "PNEUMONIA" if r["target"] == 1 else "normal"
        a1.set_title(f"{r['patientId'][:8]}  {r['view']}  truth: {truth}", fontsize=7)
        a2.set_title(f"p = {r['prob_cal']:.3f}  ({r['band']})", fontsize=7)
    fig.suptitle(title + "   [green = radiologist boxes]", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True, help="metadata.csv")
    ap.add_argument("--boxes", type=Path, default=None, help="boxes.csv (default: next to metadata.csv)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-gallery", type=int, default=6)
    args = ap.parse_args()
    if args.split == "test":
        print("NOTE: test split - only use this for the FINAL evaluation, after all choices are frozen.\n")

    cfg = json.loads((args.run / "config.json").read_text())
    cal = cu.load(args.run / "calibration_v2.json")          # per-view Platt + decision policy (step 4b)
    backbone, size = cfg["backbone"], cfg["img_size"]
    out_dir = args.run / "gradcam"
    out_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------ data (same loader as training)
    df = load_metadata(args.csv, include_not_normal=cfg.get("include_not_normal", False))
    part = df[df["split"] == args.split].reset_index(drop=True)
    boxes = pd.read_csv(args.boxes or args.csv.parent / "boxes.csv")
    boxes = boxes[boxes["patientId"].isin(part["patientId"])]
    prep = preprocess_fn(backbone)
    ds = make_dataset(df, args.split, size, args.batch_size, training=False)
    ds = ds.map(lambda x, y: prep(x), num_parallel_calls=tf.data.AUTOTUNE)

    # ------------------------------------------------------------ model + sanity check
    model = tf.keras.models.load_model(str(args.run / "best.keras"), compile=False)
    gc = GradCAM(model)
    print(f"Model: {backbone} {size}px | policy {cal['decision']['policy']} {cal['decision']['thresholds']} | "
          f"{args.split}: {len(part)} images, {int(part['target'].sum())} with opacity")

    cams, logits = [], []
    for i, xb in enumerate(ds):
        c, l = gc(xb)
        if i == 0:   # our hand-built forward pass must give the model's own output
            p_model = model(xb, training=False).numpy().ravel().astype("float64")
            diff = np.abs(p_model - 1 / (1 + np.exp(-l.astype("float64")))).max()
            print(f"Sanity check: Grad-CAM forward pass vs model output, max difference {diff:.2e}"
                  + ("  OK" if diff < 1e-2 else "  <-- PROBLEM, please send me this"))
        cams.append(c); logits.append(l)
        print(f"\r  Grad-CAM: {min((i + 1) * args.batch_size, len(part))}/{len(part)}", end="")
    print()
    cams = np.concatenate(cams)
    logits = np.concatenate(logits).astype("float64")
    part["logit"] = logits
    views = part["view"].values.astype(object)
    part["prob_cal"] = cu.probability(cal, logits, views)    # same rules as the app and step 10
    part["decision"] = cu.decide(cal, logits, views)
    part["band"] = [cu.band(cal, p) for p in part["prob_cal"]]
    part["_idx"] = np.arange(len(part))

    # ------------------------------------------------------------ scores per image
    cm = corner_mask(size)
    by_pid = dict(tuple(boxes.groupby("patientId")))
    rec = []
    for i, r in part.iterrows():
        h = upsample(cams[i], size)
        tot = h.sum()
        py, px = np.unravel_index(np.argmax(h), h.shape)
        row = dict(peak_x=px / size, peak_y=py / size,
                   cam_empty=bool(tot == 0),
                   corner_energy=float(h[cm].sum() / tot) if tot > 0 else np.nan,
                   peak_in_corner=bool(cm[py, px]) if tot > 0 else False)
        if r["target"] == 1 and r["patientId"] in by_pid:
            bm = box_mask(by_pid[r["patientId"]], size)
            row.update(box_area_frac=float(bm.mean()),
                       pointing_hit=bool(bm[py, px]) if tot > 0 else False,
                       energy_in_box=float(h[bm].sum() / tot) if tot > 0 else np.nan)
        rec.append(row)
    part = pd.concat([part, pd.DataFrame(rec)], axis=1)

    # ------------------------------------------------------------ summary
    lines = []
    def say(s=""):
        print(s); lines.append(s)

    pos = part[part["target"] == 1]
    say("=" * 74); say(f"GRAD-CAM SUMMARY — {args.split} split, {len(part)} images"); say("=" * 74)
    say(f"Empty heat-maps (no region pushes towards pneumonia): {int(part['cam_empty'].sum())} "
        f"({100 * part['cam_empty'].mean():.1f} %) - mostly confident normals, as expected")
    say()
    say("A. Does the model look INSIDE the radiologist boxes?  (pneumonia images)")
    say(f"{'group':28s}{'n':>6s}{'pointing hit':>14s}{'chance':>9s}{'energy in box':>15s}{'chance':>9s}")
    def block_a(name, d):
        if len(d) == 0:
            return
        say(f"{name:28s}{len(d):6d}{100 * d['pointing_hit'].mean():13.1f}%{100 * d['box_area_frac'].mean():8.1f}%"
            f"{100 * d['energy_in_box'].mean():14.1f}%{100 * d['box_area_frac'].mean():8.1f}%")
    block_a("all pneumonia", pos)
    block_a("  found (prob >= threshold)", pos[pos["decision"] == 1])
    block_a("  missed", pos[pos["decision"] == 0])
    block_a("  AP", pos[pos["view"] == "AP"])
    block_a("  PA", pos[pos["view"] == "PA"])
    say()
    say(f"B. Does the model look at the CORNERS (markers / text)?  uniform attention = {100 * cm.mean():.0f} %")
    say(f"{'group':28s}{'n':>6s}{'corner energy':>15s}{'peak in corner':>16s}")
    def block_b(name, d):
        d = d[~d["cam_empty"]]
        if len(d) == 0:
            return
        say(f"{name:28s}{len(d):6d}{100 * d['corner_energy'].mean():14.1f}%{100 * d['peak_in_corner'].mean():15.1f}%")
    block_b("all images", part)
    block_b("  pneumonia", pos)
    block_b("  normal", part[part["target"] == 0])
    block_b("  false alarms", part[(part["target"] == 0) & (part["decision"] == 1)])
    block_b("  AP", part[part["view"] == "AP"])
    block_b("  PA", part[part["view"] == "PA"])
    say()
    say("How to read this: pointing / energy far ABOVE chance = the model looks where the radiologists")
    say("drew boxes. Corner energy near or below 16 % = no sign of a marker/text shortcut.")

    # ------------------------------------------------------------ save + galleries
    keep = ["patientId", "class", "target", "view", "sex", "age", "logit", "prob_cal", "band", "decision",
            "peak_x", "peak_y", "cam_empty", "pointing_hit", "box_area_frac", "energy_in_box",
            "corner_energy", "peak_in_corner"]
    part[keep].to_csv(out_dir / f"gradcam_{args.split}.csv", index=False)
    (out_dir / "gradcam_summary.txt").write_text("\n".join(lines) + "\n")

    n = args.n_gallery
    fn = part[(part["target"] == 1) & (part["decision"] == 0)].sort_values("prob_cal").head(n)
    fp = part[(part["target"] == 0) & (part["decision"] == 1)].sort_values("prob_cal", ascending=False).head(n)
    tp = part[(part["target"] == 1) & (part["decision"] == 1)]
    tn = part[(part["target"] == 0) & (part["decision"] == 0)]
    tp = tp.sample(min(n, len(tp)), random_state=SEED)
    tn = tn.sample(min(n, len(tn)), random_state=SEED)
    for rows, name, title in ((fn, "confident_misses", "Most confident MISSES (pneumonia called normal)"),
                              (fp, "confident_false_alarms", "Most confident FALSE ALARMS (normal called pneumonia)"),
                              (tp, "correct_pneumonia", "Random CORRECT pneumonia"),
                              (tn, "correct_normal", "Random CORRECT normal")):
        gallery(rows, cams, boxes, 512, title, out_dir / f"gallery_{name}.png")
    print(f"\nSaved gradcam_{args.split}.csv, gradcam_summary.txt and 4 galleries in {out_dir}")


if __name__ == "__main__":
    main()
