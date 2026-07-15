"""
Testing script for VGG16 model on ImageNet with Translation (Traslación) transformations.
Evaluates model performance across a grid of horizontal and vertical shifts.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from pickle import dump
from utils import crear_mosaico, trasladar2

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

def load_dataset(csv_path, batch_size=64):
    """Helper function to load dataset from CSV."""
    df = pd.read_csv(csv_path)
    return tf.data.Dataset.from_tensor_slices((df.path, df.label_subset)) \
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size, drop_remainder=True)

# Load datasets
dst_test = load_dataset("imagenet_test.csv", 64)

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
model_VGG16.load_weights("VGG16_IMA.keras", skip_mismatch=False)

# Evaluation with Translation
CROP_SIZE = 256
MAX_SHIFT = 50
desps_h = range(-MAX_SHIFT, MAX_SHIFT + 1)
desps_v = range(-MAX_SHIFT, MAX_SHIFT + 1)

metricas = {}
total_steps = len(desps_h) * len(desps_v)
current_step = 0

print(f"Starting translation evaluation for {total_steps} combinations...")

for dh in desps_h:
    for dv in desps_v:
        current_step += 1
        if current_step % 100 == 0:
            print(f"Processing step {current_step}/{total_steps} (dh={dh}, dv={dv})")

        def apply_translation(img, label):
            # Apply mosaic padding and then translate
            img_mosaic = crear_mosaico(img)
            img_translated = trasladar2(
                img_mosaic, 
                crop=(CROP_SIZE, CROP_SIZE), 
                desp_h=dh, 
                desp_v=dv
            )
            return img_translated, label
        
        # Apply transformation to the test dataset
        dst_test_translated = dst_test.map(
            apply_translation, 
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(1)

        results = model_VGG16.evaluate(dst_test_translated, return_dict=True, verbose=0)
        metricas[(dh, dv)] = results

# Save results
output_path = "met_VGG_IMA.pkl"
with open(output_path, "wb") as f:
    dump(metricas, f)
print(f"Results saved to {output_path}")
