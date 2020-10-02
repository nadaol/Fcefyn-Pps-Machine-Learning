## -- LIBRERIAS
import tensorflow as tf
import re
import numpy as np
import os
import time
import json
from glob import glob
import pickle
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

## Codificador de imagenes : Codifica imagenes y las guarda en 'encoded_image_path'

## ----- PATHS variables

# Path para la lectura de las annotations
annotation_folder = '/workspace/datasets/COCO/annotations'
annotation_file = annotation_folder + '/captions_train2014.json'

# Path para la lectura de las imagenes
image_folder = "/workspace/datasets/COCO/train2014/"    

# Path para cargar el tokenizer
pickle_tokenizer_path = '/workspace/pickle_saves/tokenizer/tokenizer.pickle'

# Path para cargar el checkpoint
checkpoint_path = "/workspace/checkpoints/image_encoder_decoder/"  

# Path para guardar la codificacion de las imagenes (features) 
encoded_image_path = '/workspace/pickle_saves/encoded_images_eval/'

# Prefijo de las imagenes
image_prefix = 'COCO_train2014_'

all_captions = []
all_img_name_vector = []

# Retorna la imagen image_path reducida ( shape = 299,299,3 ) y normalizada para luego utilizarla como input del inceptionV3
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img) #normilize pixels [-1 , 1]
    return img, image_path

# Funcion para calcular tamaño maximo de los elementos t de tensor
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

# Limite del vocabulario a k palabras.
top_k = 5000

# Obtenemos tokenizer para dividir las captions en palabras
with open(pickle_tokenizer_path, 'rb') as handle:
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

# Cargo el modelo InceptionV3 ya entrenado con el dataset de 'imagenet'
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')

                                
new_input = image_model.input # guardo la capa input
hidden_layer = image_model.layers[-1].output  # guardo capa output

# Obtengo el modelo para usar en el codificador de imagen
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

"""
# Retorna en base al path de una imagen la misma preprocesada por inceptionV3 guardada anteriormente -- No se usa?
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy') 
  return img_tensor, cap """

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
        #print("Feature shape %s \n" % (x.shape)) #(1,64,256)
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

# Creamos objeto checkpoint para el encoder y decoder ya entrenado previamente
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)

# Establezco el checkpoint manager a usar (limite para 5 ultimos checkpoints)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# Restarauro el ultimo checkpoint de checkpoint_path, tanto para el modelo de encoder como el decoder
ckpt.restore(ckpt_manager.latest_checkpoint)


# Codifico la imagen image y la guardo en encoded_image_path .
def generate_embedding(image_path):
    
    attention_plot = np.zeros((max_length, attention_features_shape))
    image_file = image_path +  ".jpg"   # obtengo nombre el path completo de la imagen
    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image_file)[0], 0)  # expando dimension de vector de entrada
    img_tensor_val = image_features_extract_model(temp_input)  # Proceso la imagen con el InceptionV3
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    #print(img_tensor_val)
    
    features = encoder(img_tensor_val)  # aplico modelo del codificador y obtengo la codificacion de la imagen (feature)
    #print(features.shape)
    image_name = image_path.rsplit('/',1)[1]
    # Guardo la feature de la imagen en encoded_image_path .
    with open(encoded_image_path+image_name+".emb", 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

# Obtengo lista con todos los nombres de los items dentro de image_folder (lista file_arr len = (82783,)) (82783 imagenes)

#file_arr = os.listdir(image_folder)

# Lectura de annotations
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

## Cargado de captions y path de las imagenes correspondientes
all_captions = []
all_img_name_vector = []

#Sort annotations by image_id ----- agregado
annotations['annotations'] = sorted(annotations['annotations'], key = lambda i: i['image_id']) 

for annot in annotations['annotations']:  #not in order
    caption = '<start> ' + annot['caption'] + ' <end>' # Parseo annotations agregando simbolos de inicio y fin .
    image_id = annot['image_id']                        # obtengo id de la imagen correspondiente al caption
    full_coco_image_path = image_folder + image_prefix + '%012d.jpg' % (image_id) # guardo el path completo donde se encuentra la imagen correspondiente

    all_img_name_vector.append(full_coco_image_path)  # Guardo el path imagen
    all_captions.append(caption)                      # Guardo respectivo caption


# Mezclado de captions e imagenes (random_state 1)
train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)


# Limitar a num_examples captions-imagenes (414113 captions en total)(82783 images) para luego usar en el entrenamiento
#num_examples = 80000
num_examples = 80000
train_captions = train_captions [:num_examples]   # string train captions
img_name_vector = img_name_vector [:num_examples] # 

"""
# Show image by id   --- agregado

image_prefix = "COCO_train2014_"
def show_image(id):
  image = Image.open(image_folder + image_prefix + '%012d.jpg' % id )
  image.show()
"""
# Sort images name list alphabetically (same as image_id)
#file_arr = sorted(file_arr)
#file_arr = shuffle(file_arr,random_state=1)

#Split train,val dataset
TRAIN_PERCENTAGE = 0.8
train_examples = int (TRAIN_PERCENTAGE*num_examples)
img_name_train, img_name_val  = img_name_vector[:train_examples] , img_name_vector[train_examples:]

# Codifica las primeras 128 imagenes del set de evaluacion y las guarda en encoded_image_path .
max_count = 1024
counti = 0
for img_name in img_name_val:
    if (img_name[-4:]==".jpg"):           #checkea que la extensión sea jpg
        generate_embedding(img_name[:-4]) #codificacion y guardado
        counti+= 1
        #print(counti)
        if counti % 100 == 0:
            print (counti)
        if counti > max_count-1 :
            break
        

# Carga una imagen codificada por img_id   --- agregado
""" 
def load_encoded_image(img_id):
  with open(encoded_image_path + image_prefix +'%012d.emb' % (img_id) , 'rb') as handle:
    return pickle.load(handle)

encoded_image_9 = load_encoded_image(19767)
print(encoded_image_9)
""" 

