"""
Testing script for VGG16 model on ImageNet with Rotation transformations.
Uses OpenCV for rotation via tf.numpy_function.
"""

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from pickle import dump
from utils import crear_mosaico

# Set random seed
tf.keras.utils.set_random_seed(666)

def preprocess(path, label):
    """
    Load and preprocess an image from a path.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, size=(256, 256))
    return img, label

def load_dataset(csv_path):
    """Helper function to load dataset from CSV."""
    df = pd.read_csv(csv_path)
    return tf.data.Dataset.from_tensor_slices((df.path, df.label_subset)) \
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Load test dataset (without batching yet, as rotation is applied to individual images)
dst_test = load_dataset("imagenet_test.csv")

# Build VGG16 model
vgg_base = tf.keras.applications.vgg16.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(256, 256, 3),
)

for layer in vgg_base.layers:
    layer.trainable = False

model_VGG16 = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(x * 255.0)),
    vgg_base,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(160, activation="softmax")
])

# Compile and load weights
model_VGG16.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")
model_VGG16(np.ones((1, 256, 256, 3)))  # Init weights
model_VGG16.load_weights("./modelos_semilla/VGG16_IMA.keras", skip_mismatch=False)

@tf.numpy_function(Tout=tf.float32)
def rotar_np(image, crop_size, angle):
    """
    Rotate an image using OpenCV.
    """
    h, w, _ = image.shape
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
    
    # Crop to center
    start_h = (h // 2) - (crop_size // 2)
    start_w = (w // 2) - (crop_size // 2)
    cropped = rotated[start_h:start_h + crop_size, start_w:start_w + crop_size, :]
    return cropped

# Evaluation with Rotation
rotaciones = np.arange(0, 21, 1)
metricas = {}
total_rots = len(rotaciones)

for i, rot in enumerate(rotaciones):
    print(f"Evaluating rotation {i}/{total_rots}: {rot} degrees")

    def apply_rotation(img, label):
        # Create mosaic (tiled padding) using utility function
        img_mosaic = crear_mosaico(img[None, :, :, :])[0]
        # Apply rotation via numpy function
        img_rotated = rotar_np(img_mosaic, 256, rot)
        # Ensure shape for TF
        return tf.ensure_shape(img_rotated, [256, 256, 3]), label
    
    # Map rotation and batch
    dst_test_rotated = dst_test.map(
        apply_rotation, 
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(64, drop_remainder=True).prefetch(1)

    results = model_VGG16.evaluate(dst_test_rotated, return_dict=True, verbose=1)
    metricas[float(rot)] = results

# Save results
output_path = "met_VGG_rot.pkl"
with open(output_path, "wb") as f:
    dump(metricas, f)
print(f"Results saved to {output_path}")
