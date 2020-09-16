## -- LIBRERIAS
import tensorflow as tf
import re
import numpy as np
import os
import time
import json
from glob import glob
import pickle

## -----------------------------------------

annotation_folder = '/annotations/'     # directorio de annotation
annotation_file = os.path.abspath('.') +'/annotations/captions_train2014.json'  # path de annotations
image_folder = '/train2014/'            # directorio de imagenes
PATH = os.path.abspath('.') + image_folder    # path de imagenes

all_captions = []
all_img_name_vector = []

## Cargo las imagenes en vector
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3) # decodificacion en RGB
    img = tf.image.resize(img, (299, 299))     # ajusto tamanio
    img = tf.keras.applications.inception_v3.preprocess_input(img) # preprocesado con InceptionV3
    return img, image_path

# Obtener modelo InceptionV3 
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')

                                
new_input = image_model.input # guardo la capa input
hidden_layer = image_model.layers[-1].output  # guardo capa output

# maximo tamanio de cada caption en el dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

# Elijo k cantidades de palabras en el vocabulario
top_k = 5000

# Abrir archivo tokenizer
with open('./tokenizer_new/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# PARAMETROS DEL SISTEMA

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = top_k + 1
max_length = 49

# Shape del vector tomado de InceptionV3 (64, 2048)
features_shape = 2048
attention_features_shape = 64

# -------
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
# -------

# Obtengo modelo
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# Funcion de mapeo 
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy') 
  return img_tensor, cap

# Capa de atencion
class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)  # Capa dense 1
    self.W2 = tf.keras.layers.Dense(units)  # Capa dense 2
    self.V = tf.keras.layers.Dense(1)       # Capa dense 3

  def call(self, features, hidden):

    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # Aplico tanh
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # Obtengo pesos
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # Obtengo vector de contexto
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    # El encoder pasa los features a la capa FC

    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)      # Fully connected layer
        x = tf.nn.relu(x)   # Aplico RELU
        return x

class RNN_Decoder(tf.keras.Model):

  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units  
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)   # Capa embedding
    self.gru = tf.keras.layers.GRU(self.units, 
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')  # Capa GRU
    self.fc1 = tf.keras.layers.Dense(self.units)    # FC1
    self.fc2 = tf.keras.layers.Dense(vocab_size)    # FC2

    self.attention = BahdanauAttention(self.units)  # Capa de atencion

  def call(self, x, features, hidden):
    # Guardar vector de contexto y vector de pesos 
    context_vector, attention_weights = self.attention(features, hidden)

    # Capa de embedding
    x = self.embedding(x)

    # Expansion de dimensiones
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    
    # Salida de GRU
    output, state = self.gru(x)

    # Salida de FC1
    x = self.fc1(output)

    x = tf.reshape(x, (-1, x.shape[2]))

    # Salida de FC2
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    # Reseteo de vector en 0
    return tf.zeros((batch_size, self.units))

encoder = CNN_Encoder(embedding_dim)    # obtengo modelo de encoder
decoder = RNN_Decoder(embedding_dim, units, vocab_size) # obtengo modelo de decoder

optimizer = tf.keras.optimizers.Adam() # aplico modelo de Adam

# Defino path para guardar checkpoint
checkpoint_path = "./checkpoints_newnew/train"  

# Creamos checkpoints para guardar modelo y optimizador 
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)

# Establezco el checkpoint manager a usar (limite para 5 ultimos checkpoints)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# restoring the latest checkpoint in checkpoint_path
ckpt.restore(ckpt_manager.latest_checkpoint)


def generate_embedding(image):
    
    attention_plot = np.zeros((max_length, attention_features_shape)) # vacio vector
    image_file = image_folder+image+".jpg"      # obtengo nombre de la imagen

    hidden = decoder.reset_state(batch_size=1)  # vacio vector

    temp_input = tf.expand_dims(load_image(image_file)[0], 0)  # expando dimension de vector de entrada
    img_tensor_val = image_features_extract_model(temp_input)   # obtengo tensor de la imagen
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    print(img_tensor_val)
    
    features = encoder(img_tensor_val)  # aplico modelo y obtengo los features
    print(features.shape)

    # Elimino resultado de embeddings determinados      
    with open('./img_embeddings_pru/'+image+".emb", 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

image_folder = './train2014/'
import os

# Obtengo imagenes a la salida del encoder
file_arr = os.listdir(image_folder)
max_count = 10000000
counti = 0
for file_img in file_arr:
    if (file_img[-4:]==".jpg"):
        generate_embedding(file_img[:-4])
        counti+= 1
        if counti % 100 == 0:
            print (counti)
        if counti > max_count:
            break
        




