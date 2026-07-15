"""
Test VGG16 with GAP on ImageNet with rotation variations.

This script evaluates a VGG16 model with Global Average Pooling (GAP) on the ImageNet dataset,
specifically testing its performance across different image rotations.
"""

from pickle import dump

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from utils import *

# Configuration
CROP_SIZE = 256
BATCH_SIZE_TRAIN = 256 // 8
BATCH_SIZE_VAL = 256 // 4
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

# Test set is not batched here because rotation is applied before batching
dst_test = tf.data.Dataset.from_tensor_slices((imgs_test, labels_test))\
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

df_val = pd.read_csv("imagenet_val.csv")
imgs_val = df_val.path
labels_val = df_val.label_subset

dst_val = tf.data.Dataset.from_tensor_slices((imgs_val, labels_val))\
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)\
    .batch(BATCH_SIZE_VAL, drop_remainder=True)

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
final_model.load_weights("./modelos_semilla/VGG16finalGAP_IMA.keras", skip_mismatch=False)

# Rotation logic
@tf.numpy_function(Tout=tf.float32)
def rotate_image(img, crop_size, rotation_angle):
    """
    Rotate an image and crop it to the center.
    """
    h, w, _ = img.shape
    image_center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, rotation_angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, dsize=(w, h), flags=cv2.INTER_LINEAR)
    
    # Center crop
    ini_h = (h // 2) - (crop_size // 2)
    ini_w = (w // 2) - (crop_size // 2)
    result = rotated[ini_h:ini_h+crop_size, ini_w:ini_w+crop_size, :]
    return result

# Evaluation with Rotation variations
metrics = {}
rotations = np.arange(0, 21, 1)
total_rotations = len(rotations)

for i, rot in enumerate(rotations, 1):
    def apply_rotation(img, label):
        # Apply mosaic from utils (assuming it adds padding)
        img_mosaic = crear_mosaico(img[None, :, :, :])[0]
        img_rotated = rotate_image(img_mosaic, CROP_SIZE, rot)
        return tf.ensure_shape(img_rotated, [CROP_SIZE, CROP_SIZE, 3]), label
    
    dst_test_rotated = dst_test.map(apply_rotation, num_parallel_calls=tf.data.AUTOTUNE)\
        .batch(BATCH_SIZE_TEST, drop_remainder=True)\
        .prefetch(1)

    results = final_model.evaluate(dst_test_rotated, return_dict=True, verbose=1)
    metrics[rot] = results
    print(f"Processed rotation {i}/{total_rotations}: {rot} degrees")

# Save results
with open("met_VGGfinalGAP_rot.pkl", "wb") as f:
    dump(metrics, f)
