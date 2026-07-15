"""
This script evaluates the VGG16 model with GAP and Flatten layers on the ImageNet dataset
under different scaling transformations.
"""

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

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
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(256//4, drop_remainder=True)

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

# Evaluation under scaling
escalas_df = pd.read_csv("escalas_imagenet.csv")
indices = np.linspace(0, len(escalas_df) - 1, num=20, dtype=int)
escalas = escalas_df.iloc[indices, -1].to_list()

metricas = {}
total_escalas = len(escalas)

for idx, escala in enumerate(escalas, 1):
    def escalar_mosaico(data, labels, size=(256, 256)):
        """
        Resizes and pads the image symmetrically to create a mosaic effect.
        """
        scaled_img = tf.image.resize(data, size=(int(data.shape[1] * escala), int(data.shape[2] * escala)))
        
        paddings = tf.constant([[0, 0], [scaled_img.shape[1], scaled_img.shape[1]], 
                                [scaled_img.shape[2], scaled_img.shape[2]], [0, 0]])
        mosaico = tf.pad(scaled_img, paddings, mode="SYMMETRIC")
        
        while mosaico.shape[1] < min(size):
            paddings = tf.constant([[0, 0], [mosaico.shape[1], mosaico.shape[1]], 
                                    [mosaico.shape[2], mosaico.shape[2]], [0, 0]])
            mosaico = tf.pad(mosaico, paddings, mode="SYMMETRIC")

        h, w = mosaico.shape[1:3]
        ini_crop1 = (h // 2) - (size[0] // 2)
        ini_crop2 = (w // 2) - (size[1] // 2)
        cropped_img = mosaico[:, ini_crop1:ini_crop1 + size[0], ini_crop2:ini_crop2 + size[1], :]

        return cropped_img, labels

    dst_test_scaled = dst_test.map(escalar_mosaico, num_parallel_calls=tf.data.AUTOTUNE).prefetch(1)

    results = model.evaluate(dst_test_scaled, return_dict=True)
    metricas[escala] = results
    print(f"Processed scale {idx}/{total_escalas}: {escala}")

# Save results
with open("met_VGGGAPflatten_esc.pkl", "wb") as f:
    pickle.dump(metricas, f)
