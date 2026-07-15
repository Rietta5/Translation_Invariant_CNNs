"""
Training script for LPIPS-like model using VGG16 with Global Average Pooling (GAP) 
on the TID2008 dataset.
"""

import tensorflow as tf
from tensorflow.keras import layers
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from utils import Weight, PearsonCorrelation
from iqadatasets.datasets import TID2008

# --- Configuration ---
config = {
    "model": "VGGGAP",
    "batch_size": 32,
    "learning_rate": 1e-3,
    "epochs": 1500,
}

wandb.init(
    project="LPIPS_Mod",
    mode="disabled",
    job_type="training",
    config=config
)
config = wandb.config

# --- Data Loading ---
# Load TID2008 dataset from the specified directory
dataset_path = "/media/disk/vista/BBDD_video_image/Image_Quality/TID/TID2008/"
dst_train = TID2008(dataset_path).dataset

# Prepare the dataset: shuffle, batch, and map to ((img, dist), mos) format
dst_train_rdy = dst_train.shuffle(100, reshuffle_each_iteration=True) \
                         .batch(config.batch_size, num_parallel_calls=tf.data.AUTOTUNE) \
                         .map(lambda x, y, z: ((x, y), z)) \
                         .prefetch(1)

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(666)

# --- Model Definition ---
img_shape = (384, 512, 3)

# Load pre-trained VGG16 without top layers
vgg16_base = tf.keras.applications.vgg16.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=img_shape,
)

# Freeze VGG16 layers
for layer in vgg16_base.layers:
    layer.trainable = False

# Preprocessing layer for VGG16
prepro = tf.keras.layers.Lambda(
    lambda x: tf.keras.applications.vgg16.preprocess_input(
        tf.convert_to_tensor(x) * 255.0, 
        data_format=None
    )
)

# Extract intermediate Global Average Pooling outputs from specific VGG16 layers
inputs = vgg16_base.input
gap_layers = [3, 6, 10, 14, 18]
gap_outputs = [layers.GlobalAveragePooling2D()(vgg16_base.layers[i].output) for i in gap_layers]

# Create a model that outputs the intermediate GAP features
intermediate_gaps_model = tf.keras.Model(inputs, gap_outputs)
feature_extractor = tf.keras.Sequential([prepro, intermediate_gaps_model])

# Define the Siamese-like architecture for LPIPS
img_input = tf.keras.Input(shape=img_shape, name="img_input")
dist_input = tf.keras.Input(shape=img_shape, name="dist_input")

# Extract and concatenate features for both images
feat_img = layers.Concatenate(axis=-1)(feature_extractor(img_input))
feat_dist = layers.Concatenate(axis=-1)(feature_extractor(dist_input))

# Apply learnable weights to the features
weights_layer = Weight()
feat_img_weighted = weights_layer(feat_img)
feat_dist_weighted = weights_layer(feat_dist)

# Calculate the LPIPS distance (root mean square of weighted feature differences)
diff = (feat_img_weighted - feat_dist_weighted)**2
outputs = tf.keras.ops.mean(diff, axis=-1)**(0.5)

# Final model
VGGGAPLPIPS = tf.keras.Model(inputs=[img_input, dist_input], outputs=outputs)

# --- Compilation and Training ---
VGGGAPLPIPS.compile(optimizer="adam", loss=PearsonCorrelation())

checkpoint_path = "VGGGAP_IMA_LPIPS.keras"
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=25, monitor="val_loss"),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor="val_loss"),
    WandbMetricsLogger(),
    WandbModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor="val_loss")
]

history = VGGGAPLPIPS.fit(
    dst_train_rdy,
    epochs=config.epochs,
    validation_data=dst_train_rdy,
    callbacks=callbacks
)
