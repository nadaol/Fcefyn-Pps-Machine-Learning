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

## Codificador de texto : Codifica texto y lo almacena en 'encoded_captions_path'.

# Funcion para calcular tamaño maximo de los elementos t de tensor
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

## ----- PATHS variables for docker execution (mapped volume : workspace )

# Path para la lectura de las annotations
annotation_folder = '/workspace/datasets/COCO/annotations'
annotation_file = annotation_folder + '/captions_train2014.json'

# Path para cargar el checkpoint del encoder y evaluar                         
checkpoint_path = '/workspace/checkpoints/text_encoder/'

# Path para cargar el tokenizer
pickle_tokenizer_path = '/workspace/pickle_saves/tokenizer/tokenizer.pickle'

# Path para guardar la salida del codificador
encoded_captions_path = '/workspace/pickle_saves/encoded_captions/'

# Lectura de annotations 
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

#Sort annotations by image_id ----- agregado
annotations['annotations'] = sorted(annotations['annotations'], key = lambda i: i['image_id']) 

## Cargado de captions e id's de las imagenes correspondientes
all_all_captions = []
all_all_img_name_vector = []

for annot in annotations['annotations']: # annotation['annotations'][0] = {'image_id': 318556, 'id': 48, 'caption': 'A very clean and well decorated empty bathroom'}
    caption = '<start> ' + annot['caption'] + ' <end>' # Parseo annotations agregando simbolos de inicio y fin .
    all_all_captions.append(caption)                  # Guardo en caption las annotations parseadas
    all_all_img_name_vector.append(annot['image_id']) # Guardo id de la imagen correspondiente al caption (all_all_img_name_vector[0] = 318556)

print('caption : %s \nimage_id : %d \n'% (all_all_captions[0],all_all_img_name_vector[0]))

# Limite del vocabulario a k palabras.
top_k = 5000

# Obtenemos tokenizer para dividir las captions en palabras
with open(pickle_tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Obtengo la lista que representan las captions (414113 captions)
# all_all_captions[0] = <start> A very clean and well decorated empty bathroom <end>
all_all_captions = tokenizer.texts_to_sequences(all_all_captions)
# all_all_captions[0] = -> [3, 2, 136, 491, 10, 622, 430, 271, 58, 4]

all_captions = []
all_img_name_vector  = []

#Limite para la cantidad de palabras de un caption.
max_length_set = 49

# Filtro aquellas captions mayores a max_length_set
for i in range(len(all_all_captions)):
    if len(all_all_captions[i]) <= max_length_set:  
        all_captions.append(all_all_captions[i])
        all_img_name_vector.append(all_all_img_name_vector[i])

# con length 49 quedan 414108 captions
max_length = calc_max_length(all_captions)
#print(max_length)
#print(np.array(all_captions).shape)

# Aplico padding a las captions , para obtener captions con tamaño fijo = max_length
all_captions = tf.keras.preprocessing.sequence.pad_sequences(all_captions,maxlen=max_length, padding='post')

#all_captions [0] = [  3   2 136 491  10 622 430 271  58   4   0   0  ....  0   0   0   0   0   0   0   0   0   0   0   0]


# Parametros del modelo
max_length = 49
BUFFER_SIZE = 1000
BATCH_SIZE = 64
embedding_dim = 256
units = 1024
output_units = 512
output_size = 16384
vocab_inp_size = top_k + 1 # ojooo


# No tiene atencion?
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
    self.fc = tf.keras.layers.Dense(enc_output_units,act)   # FC1 - Dense ------(if no activation is specified it uses linear)
  

  def call(self, x, hidden):
    x = self.embedding(x)   # capa Embedding
    output_gru, state = self.gru(x, initial_state = hidden) # capa GRU
    cnn1out = self.cnn1(output_gru) # capa CNN
    flat_vec = tf.reshape(cnn1out,[cnn1out.shape[0],cnn1out.shape[1]*cnn1out.shape[2]]) # Flattened
    output = self.fc(flat_vec)   # capa densa de salida - FC
    output = tf.nn.relu(output) #ver si reduce el error
    return output, state

# Inicio hidden state todo en cero
  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units)) 

# Imprimir parametros de modelo de ENCODER
print("parametros encoder",vocab_inp_size, embedding_dim, units, output_size, BATCH_SIZE)

# Obtengo ENCODER
encoder = Encoder(vocab_inp_size, embedding_dim, units, output_size, BATCH_SIZE)

# Instancio objeto optimizador Adam 
optimizer = tf.keras.optimizers.Adam()

# Creamos objeto Checkpoint para cargar el encoder entrenado previamente
ckpt = tf.train.Checkpoint(encoder=encoder,
                           optimizer = optimizer)

# Establezco el checkpoint manager a usar (limite para 5 ultimos checkpoints)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# Restoring the latest checkpoint in checkpoint_path
ckpt.restore(ckpt_manager.latest_checkpoint) 

# Funcion para inicializacion y evaluación del encoder
def evaluate(caption):
  
  enc_hidden = encoder.initialize_hidden_state()
  enc_output, enc_hidden = encoder(caption, enc_hidden) 
  return enc_output

# Cargo dataset con los captions ya procesados (all_captions)
dataset = tf.data.Dataset.from_tensor_slices((all_captions,all_img_name_vector))
dataset = dataset.batch(BATCH_SIZE) # divido en batch
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # optimizado para la operacion

""" #Codifica los captions y los guarda en encoded_images_path
text_id = 0
for (batch,(caption,img_name)) in enumerate(dataset):
    text_encoded_vec = evaluate(caption) # salida del encoder
    for i in range(64): 
      text_id += 1
      full_encoded_captions_path = encoded_captions_path + 'encodedText_%012d_%012d.emdt' % (img_name[i],text_id)
      with open(full_encoded_captions_path, 'wb') as handle:
        pickle.dump(text_encoded_vec[i].numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL) 
    if batch % 20==0:
      print("batch",batch)
    if(batch == 9):       #------------- agregado para codificar solo las primeras 640 captions
      break """


# Carga una caption codificada ,por img_id y text_id  --- agregado
def load_encoded_caption(img_id,text_id):
  with open(encoded_captions_path + 'encodedText_%012d_%012d.emdt' % (img_id,text_id), 'rb') as handle:
    return pickle.load(handle)
vec = load_encoded_caption(9,1)
print(vec)
print(vec[127])

#[ 0.00625938  0.00668656  0.01346777 ...  0.00584808  0.00308381 -0.00475015]



