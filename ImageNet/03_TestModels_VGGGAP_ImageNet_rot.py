"""
Test VGGGAP model performance on ImageNet dataset under different rotations.

This script evaluates the VGGGAP model (VGG16 with intermediate Global Average Pooling layers)
against images rotated by various angles.
"""

import os
from pickle import dump

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from utils import crear_mosaico

# Configuration
CROP_SIZE = 256
BATCH_SIZE_TRAIN = 256 // 8
BATCH_SIZE_TEST = 256 // 4
MODEL_PATH = "./modelos_semilla/VGGGAP_IMA.keras"

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
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

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

# Rotation logic
@tf.numpy_function(Tout=tf.float32)
def rotate_image(img, crop_size, angle):
    """
    Rotate image using OpenCV and crop to desired size.
    
    Args:
        img: Input image.
        crop_size: Size of the final crop.
        angle: Rotation angle in degrees.
        
    Returns:
        Rotated and cropped image.
    """
    h, w, _ = img.shape
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, dsize=(w, h), flags=cv2.INTER_LINEAR)
    
    ini_h = (h // 2) - (crop_size // 2)
    ini_w = (w // 2) - (crop_size // 2)
    cropped = rotated[ini_h:ini_h + crop_size, ini_w:ini_w + crop_size, :]
    return cropped

# Evaluation under rotation
rotations = np.arange(0, 21, 1)
metrics_results = {}
total_rotations = len(rotations)

for i, rot in enumerate(rotations):
    def apply_rotation(img, label):
        """
        Create mosaic, rotate and ensure shape.
        
        Args:
            img: Input image.
            label: Image label.
            
        Returns:
            Rotated image and label.
        """
        img_mosaic = crear_mosaico(img[None, :, :, :])[0]
        img_rot = rotate_image(img_mosaic, CROP_SIZE, float(rot))
        return tf.ensure_shape(img_rot, [CROP_SIZE, CROP_SIZE, 3]), label
    
    dst_test_rotated = dst_test.map(apply_rotation, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(BATCH_SIZE_TEST, drop_remainder=True) \
        .prefetch(1)

    results = model_vgg_gap.evaluate(dst_test_rotated, return_dict=True, verbose=0)
    metrics_results[rot] = results
    print(f"Rotation {i}/{total_rotations}: {rot} degrees - Accuracy: {results['accuracy']:.4f}")

# Save results
with open("met_VGGGAP_rot.pkl", "wb") as f:
    dump(metrics_results, f)
