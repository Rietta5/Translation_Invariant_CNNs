"""
Train VGG16 models with Global Average Pooling (GAP) on translated MNIST images.
The model concatenates GAP outputs from multiple intermediate layers.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
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

    # Define the custom GAP model
    inputs = VGG16.input
    
    # Extract intermediate layers and apply Global Average Pooling
    GAPMP1 = layers.GlobalAveragePooling2D()(VGG16.layers[3].output)
    GAPMP2 = layers.GlobalAveragePooling2D()(VGG16.layers[6].output)
    GAPMP3 = layers.GlobalAveragePooling2D()(VGG16.layers[10].output)
    GAPMP4 = layers.GlobalAveragePooling2D()(VGG16.layers[14].output)
    GAPMP5 = layers.GlobalAveragePooling2D()(VGG16.layers[18].output)

    # Concatenate all GAP outputs
    GAPFinal = layers.Concatenate(axis=-1)([GAPMP1, GAPMP2, GAPMP3, GAPMP4, GAPMP5])
    outputs = layers.Dense(10, activation="softmax")(GAPFinal)

    ModeloVGGGAP_base = tf.keras.Model(inputs, outputs)

    # Preprocessing layer
    prepro = tf.keras.layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(
        tf.image.grayscale_to_rgb(tf.convert_to_tensor(x)), data_format=None
    ))

    # Final Sequential model including preprocessing
    ModeloVGGGAP = tf.keras.Sequential([prepro, ModeloVGGGAP_base])

    # Compile and train the model
    ModeloVGGGAP.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")
    history = ModeloVGGGAP.fit(
        dst_train,
        epochs=100,
        validation_data=dst_val,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5),
            tf.keras.callbacks.ModelCheckpoint(filepath=f'VGG16GAP_MNIST_{i}.keras', save_best_only=True)
        ]
    )
