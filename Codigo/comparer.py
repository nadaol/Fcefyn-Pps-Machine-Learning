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

## Compara las codificaciones de las imagenes con las de las captions ?

max_length_set = 49                     # tamanio del conjunto
annotation_folder = '/annotations/'     # directorio de annotation
image_folder = '/img_embeddings/'       # directorio de imagenes
annotation_file = os.path.abspath('.') +'/annotations/captions_train2014.json'  # path de annotations

all_all_captions = []   

# Cargo annotations
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

## Obtenemos tokenizer
with open('./tokenizer_new/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

## Guardo captions
u = 0 # contador
for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
#    print (u,caption)
    all_all_captions.append(caption)
    u+=1

# Guardo captions tokenizados
all_all_captions_tok = tokenizer.texts_to_sequences(all_all_captions)

all_captions = []
all_img_name_vector  = []

# Guardar segun tamanio
for i in range(len(all_all_captions)):
    if len(all_all_captions_tok[i]) <= max_length_set:
        all_captions.append(all_all_captions[i])

# Obtengo objeto de loss
loss_object = tf.keras.losses.MeanSquaredError()

## Cargo una unica imagen
files=glob.glob('./img_embeddings/*.emdt')
img_name = "./img_embeddings/COCO_train2014_000000493717"
img_name = img_name+'.npy'
img_tensor = np.load(img_name)

# Defino minimo tensor de texto
min_error_txt_tensor = 1000000
min_error_txt_tensor_file = "No encontre" # ???????

thebest = [] 

for i in range(len(files)):
    if (i %1000 == 0) or (i %500 == 0 and i<5000) or (i%100 == 0 and i<1000):
        print ("NUEVO")
        for s in range(len(thebest)):
            print (i,s,thebest[s])
    with open(files[i], 'rb') as handle:
       txt_tensor = pickle.load(handle) # cargo texto
    error = loss_object(txt_tensor,img_tensor)  # comparo texto e imagen
    if len(thebest) < 10:   # ?????
        thebest.append([min_error_txt_tensor,min_error_txt_tensor_file])
    else:
        for k in range(len(thebest)):   
            if error < thebest[k][0]: 
                 thebest.insert(k, [error,files[i]])
                 del thebest[10]
                 break

print(thebest) # imprimo vector final



