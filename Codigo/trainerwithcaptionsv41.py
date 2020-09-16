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

# Descarga de imagenes
image_folder = '/img_embeddings/'
PATH = os.path.abspath('.') + image_folder

# Lectura de annotations
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

## Guardo captions e imagenes
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'  # codifico captions con token inicio fin y anotations
    image_id = annot['image_id'] # obtengo id del annotation
    full_coco_image_path = PATH + 'COCO_train2014_' + '%012d' % (image_id) # guardo path con full nombre

    all_img_name_vector.append(full_coco_image_path)  # Guardo imagen
    all_captions.append(caption)                      # Guardo respectivo caption

# Mezclado de captions e imagenes 
train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)

# Seleccionar "num_examples" de datos
num_examples = 80000
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]
print (len(train_captions), len(all_captions))

# Elijo "k" palabras del vocabulario
top_k = 5000

## Obtenemos tokenizer
with open('./tokenizer_new/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Creamos vectores con tokens
train_seqs = tokenizer.texts_to_sequences(train_captions)

# Aplicamos pad a cada vector (en post)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

# Preparamos batch de imagenes, obtengo vector de imagenes
def load_image(image_path):
    img = tf.io.read_file(image_path)
    return img, image_path    
    
# Obtengo tamanio max de los train_seqs
max_length = calc_max_length(train_seqs) 

# Creamos sets de train y validacion con 80/20
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)

# Preparacion

BUFFER_SIZE = 1000
BATCH_SIZE = 64
steps_per_epoch = len(img_name_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
output_units = 512
output_size = 16384
vocab_inp_size = top_k + 1 #### OJO

## Cargo tensores de imagen
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap

# Cargo dataset con imagenes y captions con las dimensiones de cada uno
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Cargar numpy
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Mezclar y dividir en batches
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # optimizado para la operacion

## ENCODER

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
                                   recurrent_initializer='glorot_uniform') # capa GRU
    self.cnn1 = tf.keras.layers.Conv1D(256, 4, activation='relu',input_shape = [49, 1024]) # capa convolucional
    self.fc = tf.keras.layers.Dense(enc_output_units) # capa Dense

  def call(self, x, hidden):
    x = self.embedding(x) 
    output_gru, state = self.gru(x, initial_state = hidden)
    cnn1out = self.cnn1(output_gru)
    flat_vec = tf.reshape(cnn1out,[cnn1out.shape[0],cnn1out.shape[1]*cnn1out.shape[2]])
    output = self.fc(flat_vec)
    output = tf.nn.relu(output) 
    return output, state

# Inicio hidden state todo en cero
  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

# Imprimo parametros configurados del encoder
print("Parametros encoder",vocab_inp_size, embedding_dim, units, output_size, BATCH_SIZE)

# EJECUTAR ENCODER
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
ckpt = tf.train.Checkpoint(encoder = encoder,
                           optimizer = optimizer)
# Establezco el checkpoint manager a usar (limite para 5 ultimos checkpoints)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0 # contador

if ckpt_manager.latest_checkpoint: # si existen checkpoints
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1]) # 
  # restoring the latest checkpoint in checkpoint_path
  ckpt.restore(ckpt_manager.latest_checkpoint)  # cargo el ultimo checkpoint disponible                     

## METODO PARA TRAIN
@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0
  # Computo gradiente
  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)
    loss += loss_function(targ,enc_output)

  batch_loss = loss 
  variables = encoder.trainable_variables #+ decoder.trainable_variables
  gradients = tape.gradient(loss, variables) # computo gradiente 
  optimizer.apply_gradients(zip(gradients, variables))  # Aplico gradiente

  return batch_loss,enc_output  

EPOCHS = 500
print( "v41")
for epoch in range(EPOCHS):
  start = time.time() # inicio cuenta de tiempo
  enc_hidden = encoder.initialize_hidden_state() # inicio hidden state en cero
  total_loss = 0 # reinicio cuenta loss

  for (batch, (targ,cap)) in enumerate(dataset):
    #verrr
    enc_hidden = encoder.initialize_hidden_state()
    batch_loss,res = train_step(cap, targ, enc_hidden)
    print(res[50].numpy())
    print(targ[50].numpy())
    sys.exit()

    total_loss += batch_loss # computo loss de cada epoch

    if batch % 100 == 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))  

  if epoch % 2 == 0:
      ckpt_manager.save()   ## almaceno checkpoint
  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))