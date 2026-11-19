"""
classifier_utils.py — everything about the classifier MODEL that more than one script needs
============================================================================================

Imported by step 3 (training) and later by calibration, Grad-CAM, evaluation and the app,
so the backbone, the input normalisation and the layer names are defined in ONE place.

    from classifier_utils import build_model, preprocess_fn, set_trainable, LOGIT_LAYER

Why the normalisation is NOT inside the model
  Keras' own preprocess_input functions behave slightly differently between TensorFlow
  versions, and wrapping them in a Lambda layer makes saved models awkward to reload.
  So the normalisation is written out here in plain TensorFlow and applied in the data
  pipeline. The model itself only contains real layers.
"""

import tensorflow as tf

LOGIT_LAYER = "logit"   # Dense layer BEFORE the sigmoid -> needed for calibration (step 4)
PROB_LAYER = "prob"     # final sigmoid output

# ImageNet statistics the backbones were pre-trained with.
# Plain Python lists on purpose: creating a tf.constant here, at import time, would start
# TensorFlow on the GPU before the training script can set GPU memory growth.
_TORCH_MEAN = [0.485, 0.456, 0.406]       # DenseNet ("torch" mode): scale to 0-1, then z-score
_TORCH_STD = [0.229, 0.224, 0.225]
_CAFFE_MEAN = [103.939, 116.779, 123.68]  # ResNet50 ("caffe" mode): BGR, subtract mean

BACKBONES = {
    # name: (Keras constructor, normalisation mode, last conv layer for Grad-CAM, blocks to unfreeze)
    "densenet121": (tf.keras.applications.DenseNet121, "torch", "relu", ("conv5",)),
    "resnet50": (tf.keras.applications.ResNet50, "caffe", "conv5_block3_out", ("conv5",)),
}


def preprocess_fn(backbone):
    """Returns f(images) for images in 0-255, shape (..., H, W, 3) -> what the backbone expects."""
    mode = BACKBONES[backbone][1]

    def torch_mode(x):
        return (x / 255.0 - _TORCH_MEAN) / _TORCH_STD

    def caffe_mode(x):
        return x[..., ::-1] - _CAFFE_MEAN          # RGB -> BGR, subtract per-channel mean

    return torch_mode if mode == "torch" else caffe_mode


def build_model(backbone="densenet121", img_size=224, dropout=0.3, weights="imagenet"):
    """Backbone (frozen) -> global average pooling -> dropout -> Dense(1) logit -> sigmoid.

    The backbone is called with training=False. That keeps every BatchNorm layer in
    inference mode (it uses the ImageNet running mean/variance) even after we unfreeze
    the backbone in stage B. With small batches, letting BatchNorm re-estimate its
    statistics wrecks a pretrained network; this one argument prevents that.
    """
    ctor = BACKBONES[backbone][0]
    base = ctor(include_top=False, weights=weights, input_shape=(img_size, img_size, 3))
    base.trainable = False

    inp = tf.keras.Input((img_size, img_size, 3), name="image")
    x = base(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    # float32 on the last two layers: with mixed precision the rest runs in float16,
    # but the logit and probability must be float32 to be numerically safe
    logit = tf.keras.layers.Dense(1, dtype="float32", name=LOGIT_LAYER)(x)
    prob = tf.keras.layers.Activation("sigmoid", dtype="float32", name=PROB_LAYER)(logit)
    return tf.keras.Model(inp, prob, name=f"{backbone}_classifier"), base


def set_trainable(base, backbone, prefixes=None):
    """Stage B: unfreeze the layers whose name starts with one of `prefixes`
    (default: the last block, 'conv5'). BatchNorm layers always stay frozen.
    Returns the number of trainable weight tensors, to print."""
    prefixes = tuple(prefixes) if prefixes else BACKBONES[backbone][3]
    base.trainable = True
    for layer in base.layers:
        is_bn = isinstance(layer, tf.keras.layers.BatchNormalization)
        layer.trainable = (not is_bn) and (prefixes == ("all",) or layer.name.startswith(prefixes))
    return len(base.trainable_weights)


def logit_model(model):
    """Same network, but returning the logit (before sigmoid). Used for calibration."""
    return tf.keras.Model(model.input, model.get_layer(LOGIT_LAYER).output)
