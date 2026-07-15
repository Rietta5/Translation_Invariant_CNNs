"""
Training script for VGG16 model with Global Average Pooling (GAP) on the final layer.
This version uses only the last convolutional layer output for GAP.
"""

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(666)

# Configuration for Weights & Biases (Commented out in original)
# import wandb
# from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
# config = {
#      "model": "VGG16finalGAP",
#      "batch_size": 32,
#      "learning_rate": 1e-3,
#      "epochs": 1500,
# }
# wandb.init(project="ClassificationModelsInv", mode="online", job_type="training", config=config)

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
dst_train = load_dataset("imagenet_train.csv")
dst_test = load_dataset("imagenet_test.csv")
dst_val = load_dataset("imagenet_val.csv")

# Build VGG16 base model
vgg_base = tf.keras.applications.vgg16.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(256, 256, 3),
)

# Freeze convolutional layers
for layer in vgg_base.layers:
    layer.trainable = False

# Model architecture using Functional API
inputs = vgg_base.input
# Get the output of the last convolutional layer (layer 18 is block5_pool)
gap_layer = layers.GlobalAveragePooling2D()(vgg_base.layers[18].output)
outputs = layers.Dense(160, activation="softmax")(gap_layer)

modelo_vgg_gap = tf.keras.Model(inputs=inputs, outputs=outputs)

# Add preprocessing layer
prepro = tf.keras.layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(
    tf.convert_to_tensor(x) * 255.0))

final_model = tf.keras.Sequential([prepro, modelo_vgg_gap])

# Compile model
final_model.compile(
    optimizer="adam", 
    metrics=["accuracy"], 
    loss="sparse_categorical_crossentropy"
)

# Plot model summary
tf.keras.utils.plot_model(final_model, "ModeloVGGGAP_final.png")

# Training (Uncomment if needed)
# history = final_model.fit(
#     dst_train, 
#     epochs=1500, 
#     validation_data=dst_val,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(patience=25, monitor="val_accuracy"),
#         tf.keras.callbacks.ModelCheckpoint(filepath='VGG16finalGAP_IMA.keras', save_best_only=True, monitor="val_accuracy"),
#     ]
# )
