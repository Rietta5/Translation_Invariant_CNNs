"""
Test VGG16 with GAP on ImageNet with translation variations.

This script evaluates a VGG16 model with Global Average Pooling (GAP) on the ImageNet dataset,
specifically testing its performance across different image translations (horizontal and vertical).
"""

from pickle import dump

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from utils import *

# Configuration
CROP_SIZE = 256
DESPS = 50
BATCH_SIZE_TRAIN = 256 // 8
BATCH_SIZE_TEST = 256 // 4

def preprocess(path, label):
    """
    Load and preprocess an image from a path.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, size=(CROP_SIZE, CROP_SIZE))
    return img, label

# Load datasets
df_train = pd.read_csv("imagenet_train.csv")
imgs_train = df_train.path
labels_train = df_train.label_subset

dst_train = tf.data.Dataset.from_tensor_slices((imgs_train, labels_train))\
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)\
    .batch(BATCH_SIZE_TRAIN, drop_remainder=True)

df_test = pd.read_csv("imagenet_test.csv")
imgs_test = df_test.path
labels_test = df_test.label_subset

dst_test = tf.data.Dataset.from_tensor_slices((imgs_test, labels_test))\
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)\
    .batch(BATCH_SIZE_TEST, drop_remainder=True)

df_val = pd.read_csv("imagenet_val.csv")
imgs_val = df_val.path
labels_val = df_val.label_subset

dst_val = tf.data.Dataset.from_tensor_slices((imgs_val, labels_val))\
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)\
    .batch(BATCH_SIZE_TEST, drop_remainder=True)

# Model setup
input_shape = (CROP_SIZE, CROP_SIZE, 3)
VGG16_base = tf.keras.applications.vgg16.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=input_shape,
)

for layer in VGG16_base.layers:
    layer.trainable = False

inputs = VGG16_base.input
gap_layer = layers.GlobalAveragePooling2D()(VGG16_base.layers[18].output)
concat = layers.Concatenate(axis=-1)([gap_layer])
outputs = layers.Dense(160, activation="softmax")(concat)

model_vgg_gap = tf.keras.Model(inputs, outputs)

preprocess_layer = tf.keras.layers.Lambda(
    lambda x: tf.keras.applications.vgg16.preprocess_input(
        tf.convert_to_tensor(x) * 255., data_format=None
    )
)

final_model = tf.keras.Sequential([preprocess_layer, model_vgg_gap])

# Initialize model and load weights
dummy_input = np.ones((1, CROP_SIZE, CROP_SIZE, 3))
final_model(dummy_input)
final_model.compile(metrics=["accuracy"], loss="sparse_categorical_crossentropy")
# Standardizing path to match other files
final_model.load_weights("./modelos_semilla/VGG16finalGAP_IMA.keras", skip_mismatch=False)

# Evaluation with Translation variations
metrics = {}
desps_h = range(-DESPS, DESPS + 1)
desps_v = range(-DESPS, DESPS + 1)

total_steps = len(desps_h) * len(desps_v)
current_step = 0

for dh in desps_h:
    for dv in desps_v:
        current_step += 1
        def apply_translation(img, label):
            """
            Apply mosaic and translate the image.
            """
            img_mosaic = crear_mosaico(img)
            img_translated = trasladar2(img_mosaic, crop=(CROP_SIZE, CROP_SIZE), desp_h=dh, desp_v=dv)
            return img_translated, label
        
        dst_test_translated = dst_test.map(apply_translation, num_parallel_calls=tf.data.AUTOTUNE)

        results = final_model.evaluate(dst_test_translated, return_dict=True, verbose=0)
        metrics[(dh, dv)] = results
        
        if current_step % 10 == 0 or current_step == total_steps:
            print(f"Processed translation {current_step}/{total_steps}: (h={dh}, v={dv})")

# Save results
with open("met_VGGfinalGAP_IMA.pkl", "wb") as f:
    dump(metrics, f)
