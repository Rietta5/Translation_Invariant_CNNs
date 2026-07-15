"""
Test VGGGAP model performance on ImageNet dataset under translation.

This script evaluates the VGGGAP model (VGG16 with intermediate Global Average Pooling layers)
against images shifted horizontally and vertically.
"""

import os
from pickle import dump

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from utils import crear_mosaico, trasladar2

# Configuration
CROP_SIZE = 256
DESPLAZAMIENTO = 50
BATCH_SIZE_TRAIN = 256 // 8
BATCH_SIZE_TEST = 256 // 4
MODEL_PATH = "VGGGAP_IMA.keras"

def preprocess(path, label):
    """
    Basic image preprocessing for ImageNet.
    
    Args:
        path: Path to the image file.
        label: Label of the image.
        
    Returns:
        Preprocessed image and its label.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, size=(CROP_SIZE, CROP_SIZE))
    return img, label

# Load Datasets
df_train = pd.read_csv("imagenet_train.csv")
dst_train = tf.data.Dataset.from_tensor_slices((df_train.path, df_train.label_subset)) \
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE_TRAIN, drop_remainder=True)

df_test = pd.read_csv("imagenet_test.csv")
dst_test = tf.data.Dataset.from_tensor_slices((df_test.path, df_test.label_subset)) \
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE_TEST, drop_remainder=True)

df_val = pd.read_csv("imagenet_val.csv")
dst_val = tf.data.Dataset.from_tensor_slices((df_val.path, df_val.label_subset)) \
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE_TEST, drop_remainder=True)

# Build Model
vgg16_base = tf.keras.applications.vgg16.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(CROP_SIZE, CROP_SIZE, 3),
)

for layer in vgg16_base.layers:
    layer.trainable = False

# Intermediate Global Average Pooling layers
gap1 = layers.GlobalAveragePooling2D()(vgg16_base.layers[3].output)
gap2 = layers.GlobalAveragePooling2D()(vgg16_base.layers[6].output)
gap3 = layers.GlobalAveragePooling2D()(vgg16_base.layers[10].output)
gap4 = layers.GlobalAveragePooling2D()(vgg16_base.layers[14].output)
gap5 = layers.GlobalAveragePooling2D()(vgg16_base.layers[18].output)

gap_concat = layers.Concatenate(axis=-1)([gap1, gap2, gap3, gap4, gap5])
outputs = layers.Dense(160, activation="softmax")(gap_concat)

base_model = tf.keras.Model(vgg16_base.input, outputs)

# Preprocessing layer for VGG16
preprocess_input = tf.keras.layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(
    tf.convert_to_tensor(x) * 255.0, data_format=None))

model_vgg_gap = tf.keras.Sequential([preprocess_input, base_model])

# Initialize and load weights
sample_input = np.ones((1, CROP_SIZE, CROP_SIZE, 3))
model_vgg_gap(sample_input)
model_vgg_gap.compile(metrics=["accuracy"], loss="sparse_categorical_crossentropy")

if os.path.exists(MODEL_PATH):
    model_vgg_gap.load_weights(MODEL_PATH, skip_mismatch=False)
else:
    print(f"Warning: Model weights not found at {MODEL_PATH}")

# Evaluation under translation
metrics_results = {}
desps_h = range(-DESPLAZAMIENTO, DESPLAZAMIENTO + 1)
desps_v = range(-DESPLAZAMIENTO, DESPLAZAMIENTO + 1)

for desp_h in desps_h:
    for desp_v in desps_v:
        def apply_translation(img, label):
            """Apply mosaic padding and translation to the image."""
            img_mosaic = crear_mosaico(img)
            img_shifted = trasladar2(img_mosaic, crop=(CROP_SIZE, CROP_SIZE), desp_h=desp_h, desp_v=desp_v)
            return img_shifted, label

        dst_test_translated = dst_test.map(apply_translation, num_parallel_calls=tf.data.AUTOTUNE)
        
        results = model_vgg_gap.evaluate(dst_test_translated, return_dict=True, verbose=0)
        metrics_results[(desp_h, desp_v)] = results
        print(f"Translation (h={desp_h}, v={desp_v}): {results['accuracy']:.4f}")

# Save results
with open("met_VGGGAP_IMA.pkl", "wb") as f:
    dump(metrics_results, f)
