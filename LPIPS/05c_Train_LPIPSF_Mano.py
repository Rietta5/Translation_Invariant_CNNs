from tqdm.auto import tqdm

from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from pickle import dump, load
import matplotlib.pyplot as plt
import cv2
from IPython.display import clear_output
import tensorflow as tf
# import plotly.express as px
from tensorflow.keras import layers
from functools import partial
from PIL import Image
import re

from utils import *
from iqadatasets.datasets import *

import torch
import piq
from sklearn.model_selection import train_test_split

## WandB
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

config = {
     "model": "VGGGAPf",
     "batch_size": 256//8,
     "learning_rate": 1e-3,
     "epochs": 1500,
}
wandb.init(project="LPIPS_Mod",
           mode="online",
           job_type="training",
           config=config)
config = wandb.config

## Carga datos
dst_train = TID2008("../TID/TID2008/").dataset
dst_val = TID2013("../TID/TID2013/").dataset
dst_train_rdy = dst_train.shuffle(100, reshuffle_each_iteration=True)\
                         .batch(32, num_parallel_calls=tf.data.AUTOTUNE)\
                         .map(lambda x,y,z: ((x, y), z))\
                         .prefetch(1)
dst_val_rdy = dst_val.batch(32, num_parallel_calls=tf.data.AUTOTUNE)\
                     .map(lambda x,y,z: ((x, y), z))\
                     .prefetch(1)
# img, dist, mos = next(iter(dst_train_rdy))
# print(img.shape, dist.shape, mos.shape)

## Semilla aleatoria

tf.keras.utils.set_random_seed(666)

img_shape = (384,512,3)
VGG16 = tf.keras.applications.vgg16.VGG16(
include_top=False,
weights='imagenet',
input_shape=img_shape,
)

for capa in VGG16.layers:
    capa.trainable = False

prepro = tf.keras.layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(
        tf.convert_to_tensor(x)*255., data_format=None))
inputs = VGG16.input

norma1 = tf.keras.ops.sum(VGG16.layers[3].output**2, axis = (1,2), keepdims=True)**(1/2)
norma2 = tf.keras.ops.sum(VGG16.layers[6].output**2, axis = (1,2), keepdims=True)**(1/2)
norma3 = tf.keras.ops.sum(VGG16.layers[10].output**2, axis = (1,2), keepdims=True)**(1/2)
norma4 = tf.keras.ops.sum(VGG16.layers[14].output**2, axis = (1,2), keepdims=True)**(1/2)
norma5 = tf.keras.ops.sum(VGG16.layers[18].output**2, axis = (1,2), keepdims=True)**(1/2)

GAPMP1 = layers.GlobalAveragePooling2D()(VGG16.layers[3].output/(norma1+1e-6))
GAPMP2 = layers.GlobalAveragePooling2D()(VGG16.layers[6].output/(norma2+1e-6))
GAPMP3 = layers.GlobalAveragePooling2D()(VGG16.layers[10].output/(norma3+1e-6))
GAPMP4 = layers.GlobalAveragePooling2D()(VGG16.layers[14].output/(norma4+1e-6))
GAPMP5 = layers.GlobalAveragePooling2D()(VGG16.layers[18].output/(norma5+1e-6))

Out1 = VGG16.layers[3].output/(norma1+1e-6)
Out2 = VGG16.layers[6].output/(norma2+1e-6)
Out3 = VGG16.layers[10].output/(norma3+1e-6)
Out4 = VGG16.layers[14].output/(norma4+1e-6)
Out5 = VGG16.layers[18].output/(norma5+1e-6)



intermediate_gaps = tf.keras.Model(inputs, [GAPMP1, GAPMP2, GAPMP3, GAPMP4, GAPMP5, Out1, Out2, Out3, Out4, Out5])
intermediate_gaps = tf.keras.Sequential([prepro, intermediate_gaps])

img, dist = tf.keras.Input(img_shape), tf.keras.Input(img_shape)
intermediate_maps_img = intermediate_gaps(img)
intermediate_maps_dist = intermediate_gaps(dist)

canales_pesados_img = []
canales_pesados_dist=[]
for canal_img, canal_dist in zip(intermediate_maps_img[-5:], intermediate_maps_dist[-5:]):
    weights_noGAP = WeightGeneral()
    canal_pesado_img = weights_noGAP(canal_img)
    canal_pesado_img = layers.Flatten()(canal_pesado_img)
    canales_pesados_img.append(canal_pesado_img)

    canal_pesado_dist = weights_noGAP(canal_dist)
    canal_pesado_dist = layers.Flatten()(canal_pesado_dist)
    canales_pesados_dist.append(canal_pesado_dist)

canales_pesados_img = layers.Concatenate(axis=-1)(canales_pesados_img)
canales_pesados_dist = layers.Concatenate(axis=-1)(canales_pesados_dist)

intermediate_gap_img = layers.Concatenate(axis=-1)(intermediate_maps_img[:5])
intermediate_gap_dist = layers.Concatenate(axis=-1)(intermediate_maps_dist[:5])

weights = Weight()
intermediate_img = weights(intermediate_gap_img)
intermediate_dist = weights(intermediate_gap_dist)

intermediate_img = layers.Concatenate(axis=-1)([intermediate_img, canales_pesados_img])
intermediate_dist = layers.Concatenate(axis=-1)([intermediate_dist, canales_pesados_dist])

outputs = tf.keras.ops.mean((intermediate_img - intermediate_dist)**2, axis=-1)**(1/2)

VGGGAPLPIPS = tf.keras.Model([img, dist],outputs)

VGGGAPLPIPS.compile(optimizer = "adam", loss = PearsonCorrelation())

@tf.function
def train_step(model, batch):
    (img, dist), mos = batch

    with tf.GradientTape() as tape:
        pred = model((img, dist), training=True) 
        loss_value = model.compiled_loss(mos, pred)

    grads = tape.gradient(loss_value, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_value

@tf.function
def eval_step(model, batch):
    (img, dist), mos = batch
    pred = model((img, dist), training=True) 
    loss_value = model.compiled_loss(mos, pred)

    return loss_value

## Training Loop
history = {
    "epoch/loss": [],
    "epoch/val_loss": [],
}
for epoch in range(config.epochs):
    ### Train
    losses_batch = []
    for batch in dst_train_rdy:
        loss = train_step(VGGGAPLPIPS, batch)
        losses_batch.append(loss)
    history["epoch/loss"].append(np.mean(losses_batch))
    
    ### Eval
    losses_batch = []
    for batch in dst_val_rdy:
        loss = eval_step(VGGGAPLPIPS, batch)
        losses_batch.append(loss)
    history["epoch/val_loss"].append(np.mean(losses_batch))

    ### Checkpoint
    if history["epoch/val_loss"][-1] <= min(history["epoch/val_loss"]):
        # VGGGAPLPIPS.save("VGGGAPf_IMA_LPIPS.keras")
        VGGGAPLPIPS.save_weights("VGGGAPf_IMA_LPIPS.weights.h5")

    print(f'Epoch {epoch+1}: [Train] Loss: {history["epoch/loss"][-1]} [Val] Loss: {history["epoch/val_loss"][-1]}')

    wandb.log({"epoch/epoch": epoch+1,
               **{k:v[-1] for k,v in history.items()}})

wandb.finish()


# history = VGGGAPLPIPS.fit(dst_train_rdy, epochs = 1500, validation_data = dst_val_rdy,
#                             callbacks = [tf.keras.callbacks.EarlyStopping(patience=25,monitor="val_loss", mode="min"),
#                                         tf.keras.callbacks.ModelCheckpoint(filepath=f'VGGGAPf_IMA_LPIPS.keras', save_best_only=True,monitor="val_loss", mode="min"),
#                                         WandbMetricsLogger(),
#                                         # WandbModelCheckpoint(filepath="VGGGAPf_IMA_LPIPS.keras", save_best_only=True,monitor="val_loss", mode="min")
#                                         ])
