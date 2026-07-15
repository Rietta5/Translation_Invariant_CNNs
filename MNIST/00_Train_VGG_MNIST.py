"""
Train VGG16 models on translated MNIST images for different canvas sizes.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import trasladar_MNIST

# Load MNIST data
(Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.mnist.load_data()

# Preprocess data: add channel dimension
Xtrain = np.expand_dims(Xtrain, -1)
Xtest = np.expand_dims(Xtest, -1)

# Split training data into training and validation sets
Xtrain, Xval, Ytrain, Yval = train_test_split(
    Xtrain, Ytrain, test_size=10000, random_state=666
)

# Train models for different input canvas sizes
for i in [56, 128, 256]:
    # Prepare training dataset with translation
    dst_big = trasladar_MNIST(Xtrain, (i, i), 0, 0)
    dst_train = tf.data.Dataset.from_tensor_slices((dst_big, Ytrain)).batch(256 // 8, drop_remainder=True)

    # Prepare validation dataset with translation
    val = trasladar_MNIST(Xval, (i, i), 0, 0)
    dst_val = tf.data.Dataset.from_tensor_slices((val, Yval)).batch(512 // 8, drop_remainder=True)

    # Load VGG16 base model
    VGG16 = tf.keras.applications.vgg16.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(i, i, 3),
    )

    # Freeze VGG16 layers
    for capa in VGG16.layers:
        capa.trainable = False

    # Define the sequential model
    model_VGG16 = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(
            tf.image.grayscale_to_rgb(tf.convert_to_tensor(x)), data_format=None
        )),
        VGG16,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    # Compile and train the model
    model_VGG16.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")
    history = model_VGG16.fit(
        dst_train,
        epochs=100,
        validation_data=dst_val,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5),
            tf.keras.callbacks.ModelCheckpoint(filepath=f'VGG16_MNIST_{i}.keras', save_best_only=True)
        ]
    )
