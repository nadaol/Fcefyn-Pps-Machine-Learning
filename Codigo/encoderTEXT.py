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
encoded_captions_path = '/workspace/pickle_saves/encoded_captions_eval/'

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


# Limitar a num_examples captions-imagenes (414113 captions en total)(82783 images) para luego usar en el entrenamiento
#num_examples = 80000
num_examples = 120000
all_all_captions = all_all_captions[:num_examples]   # string train captions
all_all_img_name_vector = all_all_img_name_vector[:num_examples] # 

# Limite del vocabulario a k palabras.
top_k = 5000

# Obtenemos tokenizer para dividir las captions en palabras
with open(pickle_tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Obtengo la lista que representan las captions (414113 captions)
# all_all_captions[0] = <start> A very clean and well decorated empty bathroom <end>
all_all_captions = tokenizer.texts_to_sequences(all_all_captions)
# all_all_captions[0] = -> [3, 2, 136, 491, 10, 622, 430, 271, 58, 4]



""" 
all_captions = []
all_img_name_vector  = []

#Limite para la cantidad de palabras de un caption.
max_length_set = 49

 # Filtro aquellas captions mayores a max_length_set ------------------- ?
for i in range(len(all_all_captions)):
    if len(all_all_captions[i]) <= max_length_set:  
        all_captions.append(all_all_captions[i])
        all_img_name_vector.append(all_all_img_name_vector[i])  """

#maxima longitud de las captions
max_length = calc_max_length(all_all_captions)

#print(max_length)

# Aplico padding a las captions , para obtener captions con tamaño fijo = max_length
all_all_captions = tf.keras.preprocessing.sequence.pad_sequences(all_all_captions,padding='post')

#all_captions [0] = [  3   2 136 491  10 622 430 271  58   4   0   0  ....  0   0   0   0   0   0   0   0   0   0   0   0]

#caption int array to caption sentence
def cap_seq_to_string(caption_seq):
  for word_number in caption_seq:
    print("%s " % tokenizer.index_word[word_number],end='')

#Split train,val dataset
TRAIN_PERCENTAGE = 0.8
train_examples = int (TRAIN_PERCENTAGE*num_examples)
all_img_name_vector, img_name_val , all_captions, cap_val = all_all_img_name_vector[:train_examples] , all_all_img_name_vector[train_examples:] , all_all_captions[:train_examples] ,all_all_captions[train_examples:]


print("firs eval image before shuffle : ",img_name_val[0])

# Mezclado de captions e imagenes (random_state 1) train y evaluacion
#train set
all_captions, all_img_name_vector = shuffle(all_captions,all_img_name_vector,random_state=1) 
#eval set
cap_val, img_name_val = shuffle(cap_val,img_name_val,random_state=1) 

# print(all_img_name_vector[0])
# print(img_name_val[0])
print(cap_seq_to_string(cap_val[548]))
""" 
# Parametros del modelo
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
    self.fc = tf.keras.layers.Dense(enc_output_units)   # FC1 - Dense ------(if no activation is specified it uses linear)
  

  def call(self, x, hidden):
    x = self.embedding(x)   # capa Embedding
    output_gru, state = self.gru(x, initial_state = hidden) # capa GRU
    cnn1out = self.cnn1(output_gru) # capa CNN
    flat_vec = tf.reshape(cnn1out,[cnn1out.shape[0],cnn1out.shape[1]*cnn1out.shape[2]]) # Flattened
    output = self.fc(flat_vec)   # capa densa de salida - FC
    output = tf.nn.relu(output) #ver si reduce el error
    return output, state # output shape (16384,)

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
if ckpt_manager.latest_checkpoint:
        print("Checkpoint Restored from {}".format(ckpt_manager.latest_checkpoint))
else:
        print("Initializing from scratch.")

# Funcion para inicializacion y evaluación del encoder
def evaluate(caption):
  
  enc_hidden = encoder.initialize_hidden_state()
  enc_output, enc_hidden = encoder(caption, enc_hidden) 
  return enc_output

# Cargo dataset con los captions ya procesados (all_captions)
dataset = tf.data.Dataset.from_tensor_slices((cap_val,img_name_val))
dataset = dataset.batch(BATCH_SIZE) # divido en batch
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # optimizado para la operacion 
#print("caption [0] : %s \n" %cap_val[0])

for (batch,(caption,img_name)) in enumerate(dataset):
    text_encoded_vec = evaluate(caption)
    print(text_encoded_vec[0])
    break
""" 
print("------------------ Generating encoded captions and saving them in %s ----------------\n" % (encoded_captions_path) )
print("Size of eval dataset : %d \n" % len(cap_val))
print("First 3 eval images : %s \n" % (img_name_val[:3]))
""" 
#Codifica las captions del set de evalucacion y los guarda en encoded_captions_path
text_id = 0
for (batch,(caption,img_name)) in enumerate(dataset):
    text_encoded_vec = evaluate(caption) # salida del encoder
    #print("caption : %s \n encoded_caption shape : %s \n" % (caption[0],text_encoded_vec[0]))
    for i in range(64): 
      text_id += 1
      #print(("i = %d , img_name : %s \n")%(i,img_name[i]))
      full_encoded_captions_path = encoded_captions_path + 'encodedText_%012d_%012d.emdt' % (img_name[i],text_id)
      with open(full_encoded_captions_path, 'wb') as handle:
        pickle.dump(text_encoded_vec[i].numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL) 
    if batch % 10==0:
      print("batch",batch)

 # Carga una caption codificada ,por img_id y text_id  --- agregado
def load_encoded_caption(img_id,text_id):
  with open(encoded_captions_path + 'encodedText_%012d_%012d.emdt' % (img_id,text_id), 'rb') as handle:
    return pickle.load(handle)
""" 
#vec = load_encoded_caption(9,1)
#print(vec)
#print(vec[127]) 

#[ 0.00625938  0.00668656  0.01346777 ...  0.00584808  0.00308381 -0.00475015]


