"""
Test VGG16 models with Global Average Pooling (GAP) trained on MNIST across different translations.
Metrics are saved to pickle files.
"""

from pickle import dump
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from utils import trasladar_MNIST

# Load MNIST data
(_, _), (Xtest, Ytest) = tf.keras.datasets.mnist.load_data()

# Preprocess data: add channel dimension
Xtest = np.expand_dims(Xtest, -1)

# Test models for different canvas sizes and displacement ranges
for i, desps in zip([56, 128, 256], [10, 25, 50]):
    metricas = {}

    # Load VGG16 base model
    VGG16 = tf.keras.applications.vgg16.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(i, i, 3),
    )

    # Freeze VGG16 layers
    for capa in VGG16.layers:
        capa.trainable = False

    # Define the custom GAP model structure
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

    # Initialize model weights by running a dummy input
    ins = np.ones((1, i, i, 1))
    ModeloVGGGAP(ins)
    
    # Compile and load trained weights
    ModeloVGGGAP.compile(metrics=["accuracy"], loss="sparse_categorical_crossentropy")
    ModeloVGGGAP.load_weights(f"VGG16GAP_MNIST_{i}.keras", skip_mismatch=False)

    # Iterate over displacement ranges
    desps_h = range(-desps, desps + 1)
    desps_v = range(-desps, desps + 1)

    for desp_h in desps_h:
        for desp_v in desps_v:
            # Prepare test dataset with specific translation
            test = trasladar_MNIST(Xtest, (i, i), desp_h=desp_h, desp_v=desp_v)
            dst_test = tf.data.Dataset.from_tensor_slices((test, Ytest)).batch(512 // 8, drop_remainder=True)

            # Evaluate model performance
            results = ModeloVGGGAP.evaluate(dst_test, return_dict=True, verbose=0)
            metricas[(desp_h, desp_v)] = results
        
    # Save results to a pickle file
    with open(f"met_VGG16GAP_{i}.pkl", "wb") as f:
        dump(metricas, f)
