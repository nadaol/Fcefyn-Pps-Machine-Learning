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

## -------

### Tamano maximo de cada caption
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

## Descarga de annotations
annotation_folder = '/annotations/'
annotation_file = os.path.abspath('.') +'/annotations/captions_train2014.json'

max_length_set = 49

# Descarga de imagenes
image_folder = '/img_embeddings/'
PATH = os.path.abspath('.') + image_folder

# Lectura de annotations
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

## Guardo captions e imagenes
all_all_captions = []
all_all_img_name_vector = []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>' # codifico captions con token inicio fin y anottations
    all_all_captions.append(caption) # guardo caption
    all_all_img_name_vector.append(annot['image_id']) # Guardo imagen

# Elijo "k" palabras del vocabulario
top_k = 5000

## Obtenemos tokenizer
with open('./tokenizer_new/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Dividir texto en palabras
all_all_captions = tokenizer.texts_to_sequences(all_all_captions)

all_captions = []
all_img_name_vector  = []

for i in range(len(all_all_captions)):
    if len(all_all_captions[i]) <= max_length_set:
        all_captions.append(all_all_captions[i])
        all_img_name_vector.append(all_all_img_name_vector[i])

max_length = max(len(t) for t in all_captions)

# Aplico padding a las palabras
all_captions = tf.keras.preprocessing.sequence.pad_sequences(all_captions,maxlen=max_length, padding='post')

# Parametros del modelo
max_length = 49
BUFFER_SIZE = 1000
BATCH_SIZE = 64
embedding_dim = 256
units = 1024
output_units = 512
output_size = 16384
vocab_inp_size = top_k + 1 # ojooo

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, enc_output_units,batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.enc_output_units = enc_output_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) # capa Embedding
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                  return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')  # capa GRU
    self.cnn1 = tf.keras.layers.Conv1D(256, 4, activation='relu',input_shape = [49, 1024])  # capa Convolucional
    self.fc = tf.keras.layers.Dense(enc_output_units)   # FC1 - Dense 
  

  def call(self, x, hidden):
    x = self.embedding(x)   # capa Embedding
    output_gru, state = self.gru(x, initial_state = hidden) # capa GRU
    cnn1out = self.cnn1(output_gru) # capa CNN
    flat_vec = tf.reshape(cnn1out,[cnn1out.shape[0],cnn1out.shape[1]*cnn1out.shape[2]]) # Flattened
    output = self.fc(flat_vec)   # capa de salida - FC
    return output, state

# Inicio hidden state todo en cero
  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units)) 

# Imprimir parametros de modelo de ENCODER
print("parametros encoder",vocab_inp_size, embedding_dim, units, output_size, BATCH_SIZE)

# Obtengo ENCODER
encoder = Encoder(vocab_inp_size, embedding_dim, units, output_size, BATCH_SIZE)

# Agrego optimizador Adam para gradiente
optimizer = tf.keras.optimizers.Adam()

# Calculo MSE
loss_object = tf.keras.losses.MeanSquaredError()

## Funcion de perdida
def loss_function(real, pred):
  loss_ = loss_object(real, pred)
  return tf.reduce_mean(loss_)


# Defino path para guardar checkpoint                          
checkpoint_path = './ckk_pru'

# Creamos checkpoints para guardar modelo y optimizador 
ckpt = tf.train.Checkpoint(encoder=encoder,
                           optimizer = optimizer)

# Establezco el checkpoint manager a usar (limite para 5 ultimos checkpoints)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# Restoring the latest checkpoint in checkpoint_path
ckpt.restore(ckpt_manager.latest_checkpoint) 

# Funcion para inicializacion y funcionamiento de encoder
def evaluate(cap):
  
  enc_hidden = encoder.initialize_hidden_state()
  enc_output, enc_hidden = encoder(cap, enc_hidden) 
  return enc_output

# Defino path para guardar anotaciones
annotation_folder = '/annotations/'
annotation_file = os.path.abspath('.') +'/annotations/captions_train2014.json'

# Cargo dataset con imagenes y captions con las dimensiones de cada uno
dataset = tf.data.Dataset.from_tensor_slices((all_captions,all_img_name_vector))
dataset = dataset.batch(BATCH_SIZE) # divido en batch
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # optimizado para la operacion

text_id = 0
for (batch,(cap,img_name)) in enumerate(dataset):
    text_encoded_vec = evaluate(cap) # salida del encoder
    for i in range(64): 
      text_id += 1
      full_coco_text_path = PATH + 'encodedText_%012d_%012d.emdt' % (img_name[i],text_id)
      with open(full_coco_text_path, 'wb') as handle:
        # elimino elementos de texto encoded
        pickle.dump(text_encoded_vec[i].numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL) 
    if batch % 20==0:
      print("batch",batch)




