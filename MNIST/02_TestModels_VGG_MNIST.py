"""
Test VGG16 models trained on MNIST across different horizontal and vertical translations.
Metrics are saved to pickle files.
"""

from pickle import dump
import numpy as np
import tensorflow as tf
from utils import trasladar_MNIST

# Load MNIST data
(_, _), (Xtest, Ytest) = tf.keras.datasets.mnist.load_data()

# Preprocess data: add channel dimension
Xtest = np.expand_dims(Xtest, -1)

# Test models for different canvas sizes and displacement ranges
for crop, desps in zip([56, 128, 256], [10, 25, 50]):
    metricas = {}
    
    # Load VGG16 base model
    VGG16 = tf.keras.applications.vgg16.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(crop, crop, 3),
    )

    # Define the sequential model structure
    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(
            tf.image.grayscale_to_rgb(tf.convert_to_tensor(x)), data_format=None
        )),
        VGG16,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    
    # Initialize model weights by running a dummy input
    ins = np.ones((1, crop, crop, 1))
    model(ins)
    
    # Compile and load trained weights
    model.compile(metrics=["accuracy"], loss="sparse_categorical_crossentropy")
    model.load_weights(f"VGG16_MNIST_{crop}.keras", skip_mismatch=False)

    # Iterate over displacement ranges
    desps_h = range(-desps, desps + 1)
    desps_v = range(-desps, desps + 1)

    for desp_h in desps_h:
        for desp_v in desps_v:
            # Prepare test dataset with specific translation
            test = trasladar_MNIST(Xtest, (crop, crop), desp_h=desp_h, desp_v=desp_v)
            dst_test = tf.data.Dataset.from_tensor_slices((test, Ytest)).batch(512 // 8, drop_remainder=True)

            # Evaluate model performance
            results = model.evaluate(dst_test, return_dict=True, verbose=0)
            metricas[(desp_h, desp_v)] = results
        
    # Save results to a pickle file
    with open(f"met_VGG16_{crop}.pkl", "wb") as f:
        dump(metricas, f)
