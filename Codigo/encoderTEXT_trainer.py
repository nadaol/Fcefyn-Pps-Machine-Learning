

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

## Entrena el mismo modelo de encoderTEXT

# Funcion para calcular el tamaño maximo de los elementos t de tensor
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

# Path para la lectura de las annotations
annotation_folder = '/workspace/datasets/COCO/annotations'
annotation_file = annotation_folder + '/captions_train2014.json'

# Path para cargar las codificaciones de imagenes del set de entrenamiento
encoded_image_path = '/workspace/pickle_saves/encoded_images/'

# Path para la lectura de las imagenes
image_folder = "/workspace/datasets/COCO/train2014/"    
image_prefix = "COCO_train2014_"

# Path para cargar el tokenizer
pickle_tokenizer_path = '/workspace/pickle_saves/tokenizer/tokenizer.pickle'

# Defino path para guardar checkpoint del modelo encoderText
checkpoint_path = '/workspace/checkpoints/encoder_text/'

# Epochs between saves
CKPT_EPOCH_SAVE = 5
# Max checkpoints to save
CKPT_MAX = 2

# Lectura de annotations
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

## Cargado de captions y path de las imagenes correspondientes
all_captions = []
all_img_name_vector = []

#Sort annotations by image_id ----- agregado
annotations['annotations'] = sorted(annotations['annotations'], key = lambda i: i['image_id']) 

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'  # Parseo annotations agregando simbolos de inicio y fin .
    image_id = annot['image_id']                        # obtengo id de la imagen correspondiente al caption
    full_coco_image_path = encoded_image_path + image_prefix + '%012d' % (image_id) # guardo el path a las codificaciones de las imagenes (.emb)

    all_img_name_vector.append(full_coco_image_path)  # Guardo pth de la imgen
    all_captions.append(caption)                      # Guardo respectivo caption

# Limitar a num_example el set de captions-imágenes (414113 captions en total) para luego usar en el entrenamiento
num_examples = 120000
all_captions = all_captions[:num_examples]
all_img_name_vector = all_img_name_vector[:num_examples]

# Limite del vocabulario a k palabras.
top_k = 5000

# Obtenemos tokenizer para dividir las captions en palabras
with open(pickle_tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Obtengo la lista que representan las captions (num_examples captions)
# train_captions[0] = <start> A very clean and well decorated empty bathroom <end>
all_captions = tokenizer.texts_to_sequences(all_captions)
# train_captions[0] = -> [3, 2, 136, 491, 10, 622, 430, 271, 58, 4]

# Aplico padding a las captions , para obtener captions (np array , shape (80000 , 49) ) con tamaño fijo = max_length
all_captions = tf.keras.preprocessing.sequence.pad_sequences(all_captions,padding='post')

#all_captions [0] = [  3   2 136 491  10 622 430 271  58   4   0   0  ....  0   0   0   0   0   0   0   0   0   0   0   0]
    
# Obtengo tamanio max de los train_seqs (49)
max_length = calc_max_length(all_captions)  

def cap_seq_to_string(caption_seq):
  sentence = []
  for word_number in caption_seq:
    sentence.append(tokenizer.index_word[word_number])
  return sentence

# Separamos image_name_vector (paths de las imagenes) y cap_vector (captions correspondientes) para entrenamiento 80% y evaluación 20%. ver que la division sea igual al de la evaluacion (en encoderTEXT)
TRAIN_PERCENTAGE = 0.8
train_examples = int (TRAIN_PERCENTAGE*num_examples)
img_name_train, img_name_val , cap_train, cap_val = all_img_name_vector[:train_examples] , all_img_name_vector[train_examples:] , all_captions[:train_examples] , all_captions[train_examples:]

# Mezclado de captions e imagenes 
train_captions, img_name_train = shuffle(cap_train,
                                          img_name_train,
                                          random_state=1)

                                          # Mezclado de captions e imagenes 
eval_captions, img_name_eval = shuffle(cap_val,           #no se usa aca
                                          img_name_val,
                                          random_state=1)


i = 0
for (img,cap) in zip(img_name_train,train_captions):
        print("Image name : %s ," % img)
        print("Caption : %s \n " % cap_seq_to_string(cap))
        i = i + 1
        if(i==1):
                break

print("Train dataset size %d \n" % (len(train_captions)))
print("Evaluation dataset size %d \n" % (len(eval_captions)))
print("Total images in train set %d \n" % (len(set(img_name_train))) )
print("First train emb image %s \n" % img_name_train[0])
print("First eval emb image %s \n" % img_name_eval[0])


# Preparacion

BUFFER_SIZE = 1000
BATCH_SIZE = 64
steps_per_epoch = len(img_name_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
output_units = 512
output_size = 16384   # Dimensión del tensor de salida del codificador de texto
vocab_inp_size = top_k + 1 #### OJO

## Funcion ,retorna el tensor que representa una imagen (previamente guardada en formato .npy)
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.emb',allow_pickle=True) # debería entrenar con los .emb
  return img_tensor, cap

# Creo el dataset con el path de las imagenes y los captions de entrenamiento (img_name_train, cap_train)
dataset = tf.data.Dataset.from_tensor_slices((img_name_train,train_captions))

#print first 10 train set
i = 0
for (img,cap) in zip(img_name_train,train_captions):
	print("Image name : %s ," % img)
	print("Caption : %s \n " % cap_seq_to_string(cap))
	i = i + 1
	if(i==10):
		break

# Cargar numpy ya guardado de la imagen codificada en encoderIMAGE
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Mezcla el dataset y lo divide en batchses 'BATCH_SIZE' para entrenar.
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # optimizado para la operacion

example_input_batch, example_target_batch = next(iter(dataset))
print(example_input_batch.shape, example_target_batch.shape)

## Clase del modelo ENCODER (igual al encoderTEXT)

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
    self.fc = tf.keras.layers.Dense(enc_output_units,activation='relu') # capa Dense

  def call(self, x, hidden):
    x = self.embedding(x) 
    output_gru, state = self.gru(x, initial_state = hidden)
    cnn1out = self.cnn1(output_gru)
    flat_vec = tf.reshape(cnn1out,[cnn1out.shape[0],cnn1out.shape[1]*cnn1out.shape[2]])
    output = self.fc(flat_vec)
    #output = tf.nn.softmax(output)
    return output, state

# Funcion para iniciar hidden state todo en cero
  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

# Imprimo parametros configurados del encoder
print("Parametros encoder",vocab_inp_size, embedding_dim, units, output_size, BATCH_SIZE)

# instancio el modelo ENCODER
encoder = Encoder(vocab_inp_size, embedding_dim, units, output_size, BATCH_SIZE)

# Agrego optimizador Adam para gradiente
optimizer = tf.keras.optimizers.Adam()

# Creo objeto para calcular la perdida del tipo MSE (distancia euclidiana)
loss_object = tf.keras.losses.MeanSquaredError()

## Funcion de calculo de la perdida segun tensor real y predecido
def loss_function(real, pred):
  loss_ = loss_object(real, pred)
  return tf.reduce_mean(loss_)
 
# Creamos checkpoints para guardar modelo y optimizador 
ckpt = tf.train.Checkpoint(encoder = encoder,
                           optimizer = optimizer)
# Establezco el checkpoint manager a usar (limite para 3 ultimos checkpoints)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=CKPT_MAX)

start_epoch = 0

if ckpt_manager.latest_checkpoint: # si existen checkpoints
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  # restoring the latest checkpoint in checkpoint_path
  ckpt.restore(ckpt_manager.latest_checkpoint)  # cargo el ultimo checkpoint disponible  
  print("Restored from {}".format(ckpt_manager.latest_checkpoint))

else:
  print("Initializing from scratch.")

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


#entrenamiento del encoder text
EPOCHS = 100

print("\n-------------  Starting %d epoch's training for text encoder model  ------------\n"% (EPOCHS) )
print("Number of encoded images for training % d \n" % (len(sorted(set(img_name_train))))) #max 82783 images
print("Number of Input Captions %d \n" % (len(train_captions)))  # max 414k
print("Saving checkpoints every %d epochs in %s \n" % (CKPT_EPOCH_SAVE,checkpoint_path))

for epoch in range(start_epoch , EPOCHS):
  start = time.time() # inicio cuenta de tiempo
  enc_hidden = encoder.initialize_hidden_state() # inicio hidden state en cero
  total_loss = 0 # reinicio cuenta loss

  for (batch, (targ,cap)) in enumerate(dataset): #iterate over dataset (encoded image(.emb) (target), caption (input))
    #verrr
    enc_hidden = encoder.initialize_hidden_state()
    batch_loss,res = train_step(cap, targ, enc_hidden)
    #print(res[50].numpy())  #imprime un elemento cualquiera del tensor del caption con el valor real del target para comparar
    #print(targ[50].numpy())
    #sys.exit()

    total_loss += batch_loss # computo loss de cada epoch

    if batch % 100 == 0:
     print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))  

  if (epoch!=0) and (epoch % CKPT_EPOCH_SAVE ) == 0:
    ckpt_manager.save(checkpoint_number=epoch+1)   ## almaceno checkpoint cada 2 epoch's
  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
