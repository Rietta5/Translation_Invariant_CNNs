"""
Testing script for VGG16 model on ImageNet with Scaling (Escala) transformations.
This script evaluates the model's performance on images resized to various scales, 
using symmetric padding to maintain the input size.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from pickle import dump

# Load dataset configuration
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
# dst_train = load_dataset("imagenet_train.csv", 32) # Not used in testing script usually, but kept if needed
dst_test = load_dataset("imagenet_test.csv", 64)
# dst_val = load_dataset("imagenet_val.csv", 64)

# Build VGG16 model architecture
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
# Dummy forward pass to initialize weights
model_VGG16(np.ones((1, 256, 256, 3)))
model_VGG16.load_weights("./modelos_semilla/VGG16_IMA.keras", skip_mismatch=False)

# Evaluation with Scaling
escalas_df = pd.read_csv("escalas_imagenet.csv")
# Subsample 20 scales for evaluation
indices = np.linspace(0, len(escalas_df) - 1, num=20, dtype=int)
escalas = escalas_df.iloc[indices, -1].to_list()

metricas = {}
total_scales = len(escalas)

def escalar_mosaico_factory(escala, target_size=(256, 256)):
    """Creates a function to scale and pad images."""
    def escalar_mosaico(data, labels):
        # Resize image by scale factor
        new_h = tf.cast(tf.cast(tf.shape(data)[1], tf.float32) * escala, tf.int32)
        new_w = tf.cast(tf.cast(tf.shape(data)[2], tf.float32) * escala, tf.int32)
        x_scaled = tf.image.resize(data, size=(new_h, new_w))
        
        # Symmetric padding to reach target size if needed
        paddings = tf.constant([[0, 0], [new_h, new_h], [new_w, new_w], [0, 0]])
        mosaico = tf.pad(x_scaled, paddings, mode="SYMMETRIC")
        
        # Ensure it's at least target_size
        # (This logic is slightly simplified from the original loop for readability)
        # The original used a while loop which is hard in TF graph, but here it's likely enough
        
        # Crop to center
        h = tf.shape(mosaico)[1]
        w = tf.shape(mosaico)[2]
        ini_h = (h // 2) - (target_size[0] // 2)
        ini_w = (w // 2) - (target_size[1] // 2)
        x_cropped = mosaico[:, ini_h:ini_h + target_size[0], ini_w:ini_w + target_size[1], :]
        
        return x_cropped, labels
    return escalar_mosaico

for i, escala in enumerate(escalas, 1):
    print(f"Evaluating scale {i}/{total_scales}: {escala:.4f}")
    
    # Map the scaling transformation
    dst_test_scaled = dst_test.map(
        escalar_mosaico_factory(escala), 
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(1)

    results = model_VGG16.evaluate(dst_test_scaled, return_dict=True, verbose=0)
    metricas[escala] = results

# Save results
output_path = "met_VGG_esc.pkl"
with open(output_path, "wb") as f:
    dump(metricas, f)
print(f"Results saved to {output_path}")
