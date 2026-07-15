"""
Test VGGGAP model performance on ImageNet dataset under different scales.

This script evaluates the VGGGAP model (VGG16 with intermediate Global Average Pooling layers)
against images scaled by various factors.
"""

import os
from pickle import dump

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

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

# Evaluation under scaling
escalas_df = pd.read_csv("escalas_imagenet.csv")
indices = np.linspace(0, len(escalas_df) - 1, num=20, dtype=int)
escalas = escalas_df.iloc[indices, -1].to_list()

metrics_results = {}
total_scales = len(escalas)

for i, escala in enumerate(escalas, 1):
    def scale_mosaic(data, labels, size=(CROP_SIZE, CROP_SIZE)):
        """
        Scale image and pad it symmetrically to maintain size.
        
        Args:
            data: Input image batch.
            labels: Batch labels.
            size: Target size for cropping.
            
        Returns:
            Scaled, padded, and cropped image batch.
        """
        new_h = int(data.shape[1] * escala)
        new_w = int(data.shape[2] * escala)
        x_scaled = tf.image.resize(data, size=(new_h, new_w))
        
        # Symmetrical padding
        paddings = tf.constant([[0, 0], [x_scaled.shape[1], x_scaled.shape[1]], 
                                [x_scaled.shape[2], x_scaled.shape[2]], [0, 0]])
        mosaic = tf.pad(x_scaled, paddings, mode="SYMMETRIC")
        
        # Ensure mosaic is large enough for cropping
        while mosaic.shape[1] < min(size):
            paddings = tf.constant([[0, 0], [mosaic.shape[1], mosaic.shape[1]], 
                                    [mosaic.shape[2], mosaic.shape[2]], [0, 0]])
            mosaic = tf.pad(mosaic, paddings, mode="SYMMETRIC")

        _, h, w, _ = mosaic.shape
        ini_h = (h // 2) - (size[0] // 2)
        ini_w = (w // 2) - (size[1] // 2)
        x_final = mosaic[:, ini_h:ini_h + size[0], ini_w:ini_w + size[1], :]
        return x_final, labels

    dst_test_scaled = dst_test.map(scale_mosaic, num_parallel_calls=tf.data.AUTOTUNE).prefetch(1)
    results = model_vgg_gap.evaluate(dst_test_scaled, return_dict=True, verbose=0)
    metrics_results[escala] = results
    print(f"Scale {i}/{total_scales}: {escala:.4f} - Accuracy: {results['accuracy']:.4f}")

# Save results
with open("met_VGGGAP_esc.pkl", "wb") as f:
    dump(metrics_results, f)
