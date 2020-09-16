## -- LIBRERIAS
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import unicodedata
import re
import numpy as np
import os
import io
import time
import sys

import json
from glob import glob
from PIL import Image
import pickle
import glob

## -----------------------------------------

max_length_set = 49                     # tamanio del conjunto
annotation_folder = '/annotations/'     # directorio de annotation
image_folder = '/img_embeddings/'       # directorio de imagenes
annotation_file = os.path.abspath('.') +'/annotations/captions_train2014.json' # path de annotations

# Path a texto
fileTXT = './img_embeddings/encodedText_000000001232_000000043564.emdt'

# Tensor de texto
with open(fileTXT, 'rb') as handle:
    txt_tensor = pickle.load(handle)

# Defino objeto de MSE
loss_object = tf.keras.losses.MeanSquaredError()

# Cargo train
files=glob.glob('./img_embeddings/COCO_train2014_*.npy')

# Defino minimo tensor de imagen
min_error_img_tensor = 1000000
min_error_img_tensor_file = "No encontre"

thebest = []

for i in range(len(files)):
    if (i %1000 == 0) or (i %500 == 0 and i<5000) or (i%100 == 0 and i<1000):
        print ("NUEVO")
        for s in range(len(thebest)):
            print (i,s,thebest[s])
    img_tensor = np.load(files[i])  # cargo tensor

    error = loss_object(txt_tensor,img_tensor) # comparo texto e imagen

    if len(thebest) < 10:
        thebest.append([min_error_img_tensor,min_error_img_tensor_file])
    else:
        for k in range(len(thebest)):
            if error < thebest[k][0]:
                 thebest.insert(k, [error,files[i]])
                 del thebest[10]
                 break
print(thebest) # imprimo vector final



