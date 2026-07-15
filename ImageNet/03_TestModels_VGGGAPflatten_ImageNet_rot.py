"""
This script evaluates the VGG16 model with GAP and Flatten layers on the ImageNet dataset
under different rotation transformations.
"""

import pickle
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from utils import crear_mosaico

# Configure parameters
CROP_SIZE = 256
INPUT_SHAPE = (256, 256, 3)

def preprocess(path, label):
    """
    Load and preprocess an image from a path.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, size=(256, 256))
    return img, label

# Load datasets
df_train = pd.read_csv("imagenet_train.csv")
imgs_train = df_train.path
labels_train = df_train.label_subset
dst_train = tf.data.Dataset.from_tensor_slices((imgs_train, labels_train))\
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(256//8, drop_remainder=True)

df_test = pd.read_csv("imagenet_test.csv")
imgs_test = df_test.path
labels_test = df_test.label_subset
dst_test = tf.data.Dataset.from_tensor_slices((imgs_test, labels_test))\
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

df_val = pd.read_csv("imagenet_val.csv")
imgs_val = df_val.path
labels_val = df_val.label_subset
dst_val = tf.data.Dataset.from_tensor_slices((imgs_val, labels_val))\
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(256//4, drop_remainder=True)

# Build Model
vgg16_base = tf.keras.applications.vgg16.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=INPUT_SHAPE,
)

for layer in vgg16_base.layers:
    layer.trainable = False

inputs = vgg16_base.input
salida_flatten = layers.Flatten()(vgg16_base.output)

gap1 = layers.GlobalAveragePooling2D()(vgg16_base.layers[3].output)
gap2 = layers.GlobalAveragePooling2D()(vgg16_base.layers[6].output)
gap3 = layers.GlobalAveragePooling2D()(vgg16_base.layers[10].output)
gap4 = layers.GlobalAveragePooling2D()(vgg16_base.layers[14].output)
gap5 = layers.GlobalAveragePooling2D()(vgg16_base.layers[18].output)

combined_gap = layers.Concatenate(axis=-1)([gap1, gap2, gap3, gap4, gap5, salida_flatten])
outputs = layers.Dense(160, activation="softmax")(combined_gap)

base_model = tf.keras.Model(inputs, outputs)

# Preprocessing layer for VGG16
preprocess_input = tf.keras.layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(
    tf.convert_to_tensor(x) * 255.0, data_format=None))

model = tf.keras.Sequential([preprocess_input, base_model])

# Compile and load weights
model.compile(metrics=["accuracy"], loss="sparse_categorical_crossentropy")
# Initialize model with dummy input to allow weight loading
model(np.ones((1, *INPUT_SHAPE)))
model.load_weights("./modelos_semilla/VGG16GAPflatten_IMA.keras", skip_mismatch=False)

# Rotation Evaluation
@tf.numpy_function(Tout=tf.float32)
def rotar_numpy(image, crop, angle):
    """
    Rotates an image using OpenCV and crops the center.
    """
    h, w = image.shape[:2]
    image_center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, dsize=(w, h), flags=cv2.INTER_LINEAR)
    
    ini_h = (h // 2) - (crop[0] // 2)
    ini_w = (w // 2) - (crop[1] // 2)
    return rotated[ini_h:ini_h + crop[0], ini_w:ini_w + crop[1], :]

rotaciones = np.arange(0, 21, 1)
metricas = {}
total_rotaciones = len(rotaciones)

for idx, rot in enumerate(rotaciones):
    def apply_rotation(img, label):
        """
        Creates a mosaic and applies rotation.
        """
        img_mosaic = crear_mosaico(img[None, :, :, :])[0]
        rotated_img = rotar_numpy(img_mosaic, (256, 256), rot)
        return tf.ensure_shape(rotated_img, [256, 256, 3]), label
    
    dst_test_rotated = dst_test.map(apply_rotation, num_parallel_calls=tf.data.AUTOTUNE)\
        .batch(256//4, drop_remainder=True)\
        .prefetch(1)

    results = model.evaluate(dst_test_rotated, return_dict=True, verbose=1)
    metricas[rot] = results
    print(f"Processed rotation {idx + 1}/{total_rotaciones}: {rot} degrees")

# Save results
with open("met_VGGGAPflatten_rot.pkl", "wb") as f:
    pickle.dump(metricas, f)
