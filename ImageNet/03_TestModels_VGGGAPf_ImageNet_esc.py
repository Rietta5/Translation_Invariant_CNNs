"""
Test VGG16 with GAP on ImageNet with scale variations.

This script evaluates a VGG16 model with Global Average Pooling (GAP) on the ImageNet dataset,
specifically testing its performance across different image scales using a symmetric padding (mosaic) approach.
"""

from pickle import dump

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from utils import *

# Configuration
CROP_SIZE = 256
BATCH_SIZE_TRAIN = 256 // 8
BATCH_SIZE_TEST = 256 // 4

def preprocess(path, label):
    """
    Load and preprocess an image from a path.
    
    Args:
        path: Path to the image file.
        label: Label of the image.
        
    Returns:
        A tuple of (preprocessed_image, label).
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
final_model.load_weights("./modelos_semilla/VGG16finalGAP_IMA.keras", skip_mismatch=False)

# Evaluation with Scale variations
metrics = {}
scales_df = pd.read_csv("escalas_imagenet.csv")
indices = np.linspace(0, len(scales_df) - 1, num=20, dtype=int)
selected_scales = scales_df.iloc[indices, -1].to_list()

total_scales = len(selected_scales)

for i, scale in enumerate(selected_scales, 1):
    def scale_mosaic(data, labels, size=(CROP_SIZE, CROP_SIZE)):
        """
        Resize image and pad it using symmetric mode if it's smaller than the target size.
        """
        scaled_img = tf.image.resize(
            data, 
            size=(int(data.shape[1] * scale), int(data.shape[2] * scale))
        )
        
        paddings = tf.constant([
            [0, 0], 
            [scaled_img.shape[1], scaled_img.shape[1]], 
            [scaled_img.shape[2], scaled_img.shape[2]], 
            [0, 0]
        ])
        mosaic = tf.pad(scaled_img, paddings, mode="SYMMETRIC")
        
        while mosaic.shape[1] < min(size):
            paddings = tf.constant([
                [0, 0], 
                [mosaic.shape[1], mosaic.shape[1]], 
                [mosaic.shape[2], mosaic.shape[2]], 
                [0, 0]
            ])
            mosaic = tf.pad(mosaic, paddings, mode="SYMMETRIC")

        h, w = mosaic.shape[1:3]
        ini_crop1 = (h // 2) - (size[0] // 2)
        ini_crop2 = (w // 2) - (size[1] // 2)
        cropped_img = mosaic[:, ini_crop1:ini_crop1+size[0], ini_crop2:ini_crop2+size[1], :]

        return cropped_img, labels

    dst_test_scaled = dst_test.map(scale_mosaic, num_parallel_calls=tf.data.AUTOTUNE).prefetch(1)

    results = final_model.evaluate(dst_test_scaled, return_dict=True)
    metrics[scale] = results
    print(f"Processed scale {i}/{total_scales}: {scale}")

# Save results
with open("met_VGGfinalGAP_esc.pkl", "wb") as f:
    dump(metrics, f)
