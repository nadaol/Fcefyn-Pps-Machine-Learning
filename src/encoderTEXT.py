## -- LIBRERIAS
import tensorflow as tf
import re
import matplotlib.pyplot as plot
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
from pylab import rcParams

## Codificador de texto : Codifica texto y lo almacena en 'encoded_captions_path'.

# Funcion para calcular tamaño maximo de los elementos t de tensor
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

## ----- PATHS variables for docker execution (mapped volume : workspace )

# Path para la lectura de las annotations
annotation_folder = '/workspace/datasets/COCO/annotations'
annotation_file = annotation_folder + '/captions_train2014.json'

# Path para cargar el checkpoint del encoder y evaluar                         
checkpoint_path = '/workspace/checkpoints/encoder_text/'

# Path para cargar el tokenizer
pickle_tokenizer_path = '/workspace/pickle_saves/tokenizer/tokenizer.pickle'

# Path para guardar la salida del codificador
encoded_captions_path = '/workspace/pickle_saves/encoded_eval_captions/'
#encoded_captions_path = '/workspace/pickle_saves/encoded_train_captions/'

# Path para cargar las codificaciones de imagenes del set de entrenamiento
encoded_image_path = '/workspace/pickle_saves/encoded_train_images/'
# Path para cargar las codificaciones de imagenes del set de evaluacion
encoded_image_path_eval = '/workspace/pickle_saves/encoded_eval_images/'

image_prefix = "COCO_train2014_"

# Lectura de annotations 
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

#Sort annotations by image_id ----- agregado
annotations['annotations'] = sorted(annotations['annotations'], key = lambda i: i['image_id']) 

## Cargado de captions e id's de las imagenes correspondientes
all_all_captions = []
all_all_img_name_vector = []

i = 0

BATCH_SIZE = 64
num_examples = 414113
TRAIN_PERCENTAGE = 0.8
train_examples = int (TRAIN_PERCENTAGE*num_examples)
train_examples = train_examples - (train_examples % BATCH_SIZE)
eval_rest = ((num_examples-(train_examples+1)) % 64) 

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'  # Parseo annotations agregando simbolos de inicio y fin .
    image_id = annot['image_id']                        # obtengo id de la imagen correspondiente al caption
    if(i < train_examples):
      full_coco_image_path = encoded_image_path + image_prefix + '%012d' % (image_id) # guardo el path a las codificaciones de las imagenes (.emb)
    else:
      full_coco_image_path = encoded_image_path_eval + image_prefix + '%012d' % (image_id) 
    i+=1
    all_all_img_name_vector.append(full_coco_image_path)  # Guardo pth de la imgen
    all_all_captions.append(caption)                      # Guardo respectivo caption


# Limitar a num_examples captions-imagenes (414113 captions en total)(82783 images) para luego usar en el entrenamiento
#num_examples = 80000
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
  sentence = []
  for word_number in caption_seq:
    sentence.append(tokenizer.index_word[word_number])
  return sentence

#Split train,val dataset
TRAIN_PERCENTAGE = 0.8
train_examples = int (TRAIN_PERCENTAGE*num_examples)
train_examples = train_examples - (train_examples % BATCH_SIZE)
eval_rest = ((num_examples-(train_examples+1)) % 64) 
img_name_train, img_name_val ,cap_train, cap_val = all_all_img_name_vector[:train_examples] , all_all_img_name_vector[train_examples+1:(num_examples-eval_rest)] , all_all_captions[:train_examples] ,all_all_captions[train_examples+1:(num_examples-eval_rest)]

print("Caption Train set [%d - %d] \n Caption Eval set [%d - %d]"%(0,train_examples,train_examples+1,(num_examples-eval_rest)))

#Captions set to encode
captions_to_encode = cap_val
correlated_image_names = img_name_val

# print(all_img_name_vector[0])
# print(img_name_val[0])
print("First caption after shuffle : %s \n" % (cap_seq_to_string(captions_to_encode[0])))
print("Correlated image : %s \n" , img_name_val[0])
# Parametros del modelo
BUFFER_SIZE = 1000
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
ckpt_manager = tf.train.CheckpointManager(ckpt,checkpoint_path, max_to_keep=10)

""" 
# Funcion de evaluación del encoder por batches --------------------------------------------------------

## Funcion ,retorna el tensor que representa una imagen (previamente guardada en formato .npy)

                                          # Mezclado de captions e imagenes 
eval_captions, img_name_eval = shuffle(cap_val,           #no se usa aca
                                          img_name_val,
                                          random_state=1)

def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.emb',allow_pickle=True) # debería entrenar con los .emb
  return img_tensor, cap

# Creo objeto para calcular la perdida del tipo MSE (distancia euclidiana)
loss_object = tf.keras.losses.MeanSquaredError()

#Creo dataset con el path de las imagenes y los captions de evaluacion
dataset_eval = tf.data.Dataset.from_tensor_slices((img_name_eval,eval_captions))

# Mapear dataset con las imagenes codificadas y las captions de evaluacion
dataset_eval = dataset_eval.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)      

dataset_eval = dataset_eval.batch(BATCH_SIZE) # divido en batch
dataset_eval = dataset_eval.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # optimizado para la operacion     

## Funcion de calculo de la perdida segun tensor real y predecido
def loss_function(real, pred):
  loss_ = loss_object(real, pred)
  return tf.reduce_mean(loss_)

def get_batch_loss(captions,encoded_images):
  enc_hidden = encoder.initialize_hidden_state()
  enc_output, enc_hidden = encoder(captions, enc_hidden)
  loss = 0
  for i in range (BATCH_SIZE):
    loss += loss_function(enc_output[i],encoded_images[i])
  return loss/BATCH_SIZE
  
def get_avg_loss():
  loss = 0
  for (batch,(targ,cap) ) in enumerate(dataset_eval):
    loss+=get_batch_loss(cap,targ)
  return (loss/batch)

def get_eval_losses():
  eval_loss = []
  eval_loss_epochs = []
  print("------- Starting evaluation of encoder text checkpoints -------\n")
  eval_loss.append(float(get_avg_loss()))
  eval_loss_epochs.append(0)
  print("Evaluation loss epoch (%d) : %s \n" % (eval_loss_epochs[0],eval_loss[0]) )
  for i in range(10):
    checkpoint_name = checkpoint_path + 'ckpt-' + str(10*(i+1))
    tf.print("----------- Restoring from {} -----------".format(checkpoint_name))
    ckpt.restore(checkpoint_name)
    eval_loss.append(float(get_avg_loss()))
    eval_loss_epochs.append(int(10*(i+1)))
    print("Evaluation loss epoch (%d) : %s \n" % (eval_loss_epochs[i+1],eval_loss[i+1]) )
  return eval_loss_epochs,eval_loss

eval_losses_epochs,eval_losses = get_eval_losses()

# ------------------------------------------------------------------------------ Plot


# Plot training losses from log and the evalutaion losses of encoder text checkpoints

# PATHS
file_name = "encoder_text_100epoch.txt"
log_path = "./Evaluations/logs/"
file_path = log_path + file_name

# Abrir y leer log.txt

numEpoch = []       # lista de epochs
train_Loss = []        # train_loss

f = open(file_path, "r")
linea = f.readlines()

indice = 0
for renglon in linea:
        Train_line = re.search(r"Epoch\s(\d+)\sLoss\s(\d+.\d+)", renglon)
        if Train_line is not None:
            numEpoch.append( int(Train_line.group(1)) )
            train_Loss.append( float(Train_line.group(2)) )
min_train_loss = 100.0
#get min train loss
for i in range(len(numEpoch)):
    if(train_Loss[i] < min_train_loss):
        min_train_loss = train_Loss[i]
        min_train_epoch = i
#   print("numEpoch: %d -- Train Loss: %f Evalutaion Loss : %f \n" % (numEpoch[i], train_Loss[i],eval_Loss[i]))

print("Minimum training loss at epoch %d : %f\n" % (min_train_epoch,min_train_loss) )

## Generar grafico y mostrar

plot.plot(numEpoch, train_Loss,'go--',label='Training Loss')
plot.plot(eval_losses_epochs,eval_losses,'ro--',label='Evalutaion Loss')
plot.legend(loc='lower left')
plot.xlabel("Epoch")
plot.ylabel("Losses")
plot.title("Losses in encoder text training")
plot.rcParams["figure.figsize"] = (15,10)
plot.savefig(log_path + 'encoder_text_loss_plot.png')
print("Plot saved succesfully in %s \n" % (log_path) )


# ---------------------------------------------------------------------------------------------------------------------

 """

#Restore specific checkpoint
ckpt.restore(checkpoint_path+'ckpt-10')

""" if ckpt_manager.latest_checkpoint: # si existen checkpoints
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  # restoring the latest checkpoint in checkpoint_path
  ckpt.restore(ckpt_manager.latest_checkpoint)  # cargo el ultimo checkpoint disponible  
  print("Restored from {}".format(ckpt_manager.latest_checkpoint))

else:
  print("Initializing from scratch.") """


 # Funcion para inicializacion y evaluación del encoder
def evaluate(caption):
  
  enc_hidden = encoder.initialize_hidden_state()
  enc_output, enc_hidden = encoder(caption, enc_hidden) 
  return enc_output

# Creo el dataset con captions_to_encode y correlated_image_names
dataset = tf.data.Dataset.from_tensor_slices((captions_to_encode,correlated_image_names))
dataset = dataset.batch(BATCH_SIZE) # divido en batch
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # optimizado para la operacion 

caption_batch,correlated_img_batch = next(iter(dataset))

print("------------------ Generating encoded captions and saving them in %s ----------------\n" % (encoded_captions_path) )
print("Size of the set of captions to encode : %d \n" % len(captions_to_encode))
print("First captions to encode : %s , correlated image %s \n" % (cap_seq_to_string(np.array(caption_batch[0])),correlated_img_batch[0]) )

#Codifica las captions de captions_to_encode y los guarda en encoded_captions_path
text_id = 0
for (batch,(caption,img_name)) in enumerate(dataset):
    encoded_captions_batch = evaluate(caption) # salida del encoder
    #print("caption : %s \n encoded_caption shape : %s \n" % (caption[0],text_encoded_vec[0]))
    for i in range(64): 
      text_id += 1
      #print(("i = %d , img_name : %s \n")%(i,img_name[i]))
      image_name = str(img_name[i])
      img_id = int(image_name.rsplit('/',1)[1].split('_')[2][:12])
      full_encoded_captions_path = encoded_captions_path + 'encodedText_%012d_%012d.emdt' % (img_id,text_id)
      with open(full_encoded_captions_path, 'wb') as handle:
        pickle.dump(encoded_captions_batch[i].numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL) 
    if batch % 10==0:
      print("batch",batch)
      
 # Carga una caption codificada ,por img_id y text_id  --- agregado
def load_encoded_caption(img_id,text_id):
  with open(encoded_captions_path + 'encodedText_%012d_%012d.emdt' % (img_id,text_id), 'rb') as handle:
    return pickle.load(handle)
 
#vec = load_encoded_caption(9,1)
#print(vec)
#print(vec[127]) 

#[ 0.00625938  0.00668656  0.01346777 ...  0.00584808  0.00308381 -0.00475015]
