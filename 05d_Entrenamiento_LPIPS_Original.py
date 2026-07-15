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
     "model": "VGGGAP",
     "batch_size": 256//8,
     "learning_rate": 1e-3,
     "epochs": 1500,
}
wandb.init(project="LPIPS_Mod",
           name="LPIPS_Original_NoBias",
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

def normalize_tensor(tensor):
    norm = tf.keras.ops.sum(tensor**2, axis=-1, keepdims=True)**(1/2)
    return tensor/(norm+1e-6)

Out1 = VGG16.layers[3].output
Out2 = VGG16.layers[6].output
Out3 = VGG16.layers[10].output
Out4 = VGG16.layers[14].output
Out5 = VGG16.layers[18].output

Norm1 = normalize_tensor(Out1)
Norm2 = normalize_tensor(Out2)
Norm3 = normalize_tensor(Out3)
Norm4 = normalize_tensor(Out4)
Norm5 = normalize_tensor(Out5)

intermediate_gaps = tf.keras.Model(inputs, [Norm1, Norm2, Norm3, Norm4, Norm5])
intermediate_gaps = tf.keras.Sequential([prepro, intermediate_gaps])

img, dist = tf.keras.Input(img_shape), tf.keras.Input(img_shape)

## Intermediates de imagen y distorsionada
intermediate_img = intermediate_gaps(img)
intermediate_dist = intermediate_gaps(dist)

## Diffs
diffs = [(im-di)**2 for im,di in zip(intermediate_img, intermediate_dist)]

## Combinación lineal de los canales
diffs = [layers.Conv2D(1, kernel_size=(1,1), use_bias=False)(di) for di in diffs]

## Media espacial
diffs = [tf.keras.ops.mean(di, axis=(1,2)) for di in diffs]

## Sumar todo
diffs = layers.Concatenate(axis=-1)(diffs)
diffs = tf.keras.ops.sum(diffs, axis=1)

VGGGAPLPIPS = tf.keras.Model([img, dist],diffs)

VGGGAPLPIPS.compile(optimizer = "adam", loss = PearsonCorrelation())
history = VGGGAPLPIPS.fit(dst_train_rdy, epochs = 1500, validation_data = dst_val_rdy,
                            callbacks = [
                                        # tf.keras.callbacks.EarlyStopping(patience=25,monitor="val_loss", mode="min"),
                                        tf.keras.callbacks.ModelCheckpoint(filepath=f'VGGGAP_IMA_LPIPS_Original_NoBias.keras', save_best_only=True,monitor="val_loss", mode="min"),
                                        WandbMetricsLogger(),
                                        # WandbModelCheckpoint(filepath="VGGGAP_IMA_LPIPS_Original.keras", save_best_only=True,monitor="val_loss", mode="min"),
                                        ])
