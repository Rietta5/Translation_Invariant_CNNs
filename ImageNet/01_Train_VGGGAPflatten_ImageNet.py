"""
Training script for VGG16 model with Multi-layer Global Average Pooling (GAP) and Flattened output.
This version concatenates GAP outputs from multiple convolutional blocks with the final flattened output.
"""

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(666)

# Configuration for Weights & Biases
config = {
    "model": "VGG16GAPflatten",
    "batch_size": 32,
    "learning_rate": 1e-3,
    "epochs": 1500,
}

wandb.init(
    project="ClassificationModelsInv",
    mode="online",
    job_type="training",
    config=config
)
config = wandb.config

def preprocess(path, label):
    """
    Load and preprocess an image from a path.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, size=(256, 256))
    return img, label

def load_dataset(csv_path, batch_size=32):
    """Helper function to load dataset from CSV."""
    df = pd.read_csv(csv_path)
    return tf.data.Dataset.from_tensor_slices((df.path, df.label_subset)) \
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size, drop_remainder=True)

# Load datasets
dst_train = load_dataset("imagenet_train.csv", config.batch_size)
dst_test = load_dataset("imagenet_test.csv", config.batch_size)
dst_val = load_dataset("imagenet_val.csv", config.batch_size)

# Build VGG16 base model
vgg_base = tf.keras.applications.vgg16.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(256, 256, 3),
)

# Freeze convolutional layers
for layer in vgg_base.layers:
    layer.trainable = False

# Model architecture with multi-layer GAP and Flatten
inputs = vgg_base.input

# Flatten the final output
final_flatten = layers.Flatten()(vgg_base.output)

# GAP from multiple blocks
gap1 = layers.GlobalAveragePooling2D()(vgg_base.layers[3].output)
gap2 = layers.GlobalAveragePooling2D()(vgg_base.layers[6].output)
gap3 = layers.GlobalAveragePooling2D()(vgg_base.layers[10].output)
gap4 = layers.GlobalAveragePooling2D()(vgg_base.layers[14].output)
gap5 = layers.GlobalAveragePooling2D()(vgg_base.layers[18].output)

# Concatenate all GAP outputs plus the final flatten output
combined_features = layers.Concatenate(axis=-1)([gap1, gap2, gap3, gap4, gap5, final_flatten])

# Final classifier
outputs = layers.Dense(160, activation="softmax")(combined_features)

modelo_vgg_gap_flatten = tf.keras.Model(inputs=inputs, outputs=outputs)

# Preprocessing lambda
prepro = tf.keras.layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(
    tf.convert_to_tensor(x) * 255.0))

final_model = tf.keras.Sequential([prepro, modelo_vgg_gap_flatten])

# Compile model
final_model.compile(
    optimizer="adam", 
    metrics=["accuracy"], 
    loss="sparse_categorical_crossentropy"
)

# Training
history = final_model.fit(
    dst_train, 
    epochs=config.epochs, 
    validation_data=dst_val,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=25, monitor="val_accuracy"),
        tf.keras.callbacks.ModelCheckpoint(filepath='VGG16GAPflatten_IMA.keras', save_best_only=True, monitor="val_accuracy"),
        WandbMetricsLogger(),
        WandbModelCheckpoint(filepath="VGG16GAPflatten_IMA.keras", save_best_only=True, monitor="val_accuracy")
    ]
)
