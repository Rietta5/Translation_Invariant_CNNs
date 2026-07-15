"""
Training script for VGG16 model on ImageNet subset.
This script initializes a VGG16 model, freezes its convolutional layers,
and trains a new dense classifier on top.
"""

import pandas as pd
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(666)

# Configuration for Weights & Biases
config = {
    "model": "VGG",
    "batch_size": 32,  # 256//8
    "learning_rate": 1e-3,
    "epochs": 25,
    "dropout": 0,
    "L2": 1e-4
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
    
    Args:
        path: Path to the image file.
        label: Integer label for the image.
        
    Returns:
        A tuple of (preprocessed_image, label).
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, size=(256, 256))
    return img, label

# Load datasets
def load_dataset(csv_path, batch_size):
    """Helper function to load dataset from CSV."""
    df = pd.read_csv(csv_path)
    imgs = df.path
    labels = df.label_subset
    return tf.data.Dataset.from_tensor_slices((imgs, labels)) \
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size, drop_remainder=True)

batch_size = config.batch_size
dst_train = load_dataset("imagenet_train.csv", batch_size)
dst_test = load_dataset("imagenet_test.csv", batch_size)
dst_val = load_dataset("imagenet_val.csv", batch_size)

# Build VGG16 base model
vgg_base = tf.keras.applications.vgg16.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(256, 256, 3),
)

# Freeze convolutional layers
for layer in vgg_base.layers:
    layer.trainable = False

# Define the complete model
model_VGG16 = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(x * 255.)),
    vgg_base,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        160, 
        kernel_regularizer=tf.keras.regularizers.L2(l2=config.L2), 
        activation="softmax"
    )
])

# Compile model
model_VGG16.compile(
    optimizer="adam", 
    metrics=["accuracy"], 
    loss="sparse_categorical_crossentropy"
)

# Build and plot model summary
model_VGG16.build((None, 256, 256, 3))
tf.keras.utils.plot_model(model_VGG16, "model_VGG16.png")

# Training (Uncomment if needed)
# history = model_VGG16.fit(
#     dst_train, 
#     epochs=config.epochs, 
#     validation_data=dst_val,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(patience=25, monitor="val_accuracy"),
#         tf.keras.callbacks.ModelCheckpoint(filepath='VGG_IMA_L2.keras', save_best_only=True, monitor="val_accuracy"),
#         WandbMetricsLogger(),
#     ]
# )
