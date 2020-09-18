## -- LIBRERIAS
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
import sys

## Entrena el modelo de encoderIMAGE

# Path para la lectura de las annotations
annotation_folder = '/workspace/datasets/COCO/annotations'
annotation_file = annotation_folder + '/captions_train2014.json'

# Path para la lectura de las imagenes
image_folder = "/workspace/datasets/COCO/train2014/"  

# Path para guardar las imagenes preprocesadas con InceptionV3
prepro_images_folder = "/workspace/datasets/COCO/train2014_nptensors/"  

# Prefijo de las imagenes
image_prefix = 'COCO_train2014_'

# Path para guardar el tokenizer
pickle_tokenizer_path = '/workspace/pickle_saves/tokenizer/tokenizer.pickle'

# Path para guardar los checkpoints del modelo 
checkpoint_path = "/workspace/checkpoints/image_encoder_decoder/"  

# Descargo annotations
## Si no existe
if not os.path.exists(annotation_file):
  annotation_zip = tf.keras.utils.get_file('captions.zip',
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                          extract = True)
  os.remove(annotation_zip)

if not os.path.exists(image_folder): # si no existe path con imagenes, descargo
  image_zip = tf.keras.utils.get_file('train2014.zip',
                                      cache_subdir=os.path.abspath('.'),
                                      origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                      extract = True)
  os.remove(image_zip)

# Lectura de annotations
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

## Cargado de captions y path de las imagenes correspondientes
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>' # Parseo annotations agregando simbolos de inicio y fin .
    image_id = annot['image_id']                        # obtengo id de la imagen correspondiente al caption
    full_coco_image_path = image_folder + image_prefix + '%012d.jpg' % (image_id) # guardo el path completo donde se encuentra la imagen correspondiente

    all_img_name_vector.append(full_coco_image_path)  # Guardo el path imagen
    all_captions.append(caption)                      # Guardo respectivo caption

# Mezclado de captions e imagenes 
train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)

# Limitar a num_examples captions-imagenes (414113 captions en total) para luego usar en el entrenamiento
#num_examples = 80000
num_examples = 10
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]
print (len(train_captions), len(all_captions))

# Retorna la imagen image_path reducida ( shape = 299,299,3 ) y normalizada para luego utilizarla como input del inceptionV3
def load_image(image_path):
    img_preprocessed = tf.io.read_file(image_path)
    img_preprocessed = tf.image.decode_jpeg(img_preprocessed, channels=3) # decodificacion en RGB
    img_preprocessed = tf.image.resize(img_preprocessed, (299, 299)) # ajusto tamanio
    img_preprocessed = tf.keras.applications.inception_v3.preprocess_input(img_preprocessed) # preprocesado con InceptionV3
    return img_preprocessed , image_path

# Cargo el modelo InceptionV3 ya entrenado con el dataset de 'imagenet'
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')

new_input = image_model.input   # guardo la capa input
hidden_layer = image_model.layers[-1].output # guardo capa output

# Obtengo el modelo para usar en el codificador de imagen
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# Obtengo path de imagenes en orden ascendente por nombre (igual que ordenar por image_id) ... ordena lo que había mezclado ...
encode_train = sorted(set(img_name_vector))

# Creo el dataset con el path de las imagenes ordenado
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
# Divide el dataset por batches
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)


#Obtiene codificación de la imagen y le hace el reshape (no habíamos obtenido ya la codificaciones de las imagenes con encoderIMAGE ?)
for img, path in image_dataset:
  batch_features = image_features_extract_model(img)#Toma las imagenes presprocesadas y las pasa por el inceptionV3
  batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3])) # cambio dimension

  for bf, p in zip(batch_features, path):
    path_of_feature = p.numpy().decode("utf-8") 
    np.save(path_of_feature, bf.numpy())  #guardo batch features en un zip


# Funcion para calcular el tamaño maximo de los elementos t de tensor
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

# Limite del vocabulario a k palabras.
top_k = 5000
# Obtenemos tokenizer para dividir las captions en palabras
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)

tokenizer.word_index['<pad>'] = 0     # defino valor para pad
tokenizer.index_word[0] = '<pad>'

# Guardo el tokenizer creado
with open(pickle_tokenizer_path , 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Crear vectores tokenizados
train_seqs = tokenizer.texts_to_sequences(train_captions) # obtengo secuencia de enteros que matchean las captions (num_examples captions)...segunda vez

# Aplico padding a las captions , para obtener captions (np array , shape (80000 , 49) ) con tamaño fijo = max_length
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

# Obtengo tamanio max de los train_seqs (49)
max_length = calc_max_length(train_seqs)

# Separacion de conjuntos de entrenamiento y prueba (ver que el conjunto que la division coincida cuando se evalua el modelo en encoderIMAGE
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)

print(len(img_name_train), len(cap_train), len(img_name_val), len(cap_val))

# PARAMETROS DEL MODELO

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = top_k + 1
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
features_shape = 2048
attention_features_shape = 64

# Cargo elementos de numpy
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap

# Creo el dataset con el path de las imagenes y los captions de entrenamiento (img_name_train, cap_train)
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Carga en paralelo
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Mezcla el dataset y lo divide en batchses 'BATCH_SIZE' para entrenar.
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

#Modelo del encoder-decoder de imagenes-texto

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units) # FC1
    self.W2 = tf.keras.layers.Dense(units) # FC2
    self.V = tf.keras.layers.Dense(1) # FC3

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
        x = self.fc(x)    # Fully connected layer
        x = tf.nn.relu(x) # Aplico RELU
        return x


class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) # Capa embedding
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')  # Capa GRU
    self.fc1 = tf.keras.layers.Dense(self.units) # FC1
    self.fc2 = tf.keras.layers.Dense(vocab_size) # FC2

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

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # Salida de FC2
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
        # Reseteo de vector en 0
    return tf.zeros((batch_size, self.units))

encoder = CNN_Encoder(embedding_dim)  # obtengo modelo de encoder
decoder = RNN_Decoder(embedding_dim, units, vocab_size) # obtengo modelo de decoder

optimizer = tf.keras.optimizers.Adam()  # aplico modelo de Adam

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(  
    from_logits=True, reduction='none')  ## UTILIZO CROSS ENTROPY

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)   # Obtengo loss entre lo obtenido y lo esperado

  mask = tf.cast(mask, dtype=loss_.dtype) 
  loss_ *= mask

  return tf.reduce_mean(loss_)

ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer) # Creamos checkpoints para guardar modelo y optimizador 

ckpt_manager = tf.train.CheckpointManager(ckpt, 
              checkpoint_path, max_to_keep=5) 
# Establezco el checkpoint manager a usar (limite para 5 ultimos checkpoints)


start_epoch = 0 # contador

if ckpt_manager.latest_checkpoint: # si existen checkpoints
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  # restoring the latest checkpoint in checkpoint_path
  ckpt.restore(ckpt_manager.latest_checkpoint)  # cargo el ultimo checkpoint disponible  


loss_plot = []
@tf.function
def train_step(img_tensor, target):
  loss = 0

  # Inicializacion del hidden state en cada batch debido a que los captions no estan relacionados
  # entre una imagen y otra
  hidden = decoder.reset_state(batch_size=target.shape[0])

  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1) # expando dimension

  with tf.GradientTape() as tape:   # Computo gradiente
      features = encoder(img_tensor)  # Obtengo los features a la salida del encoder

      for i in range(1, target.shape[1]):
          # Paso los features al decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)
          # Computo la suma de loss
          loss += loss_function(target[:, i], predictions)
          # Usando teacher forcing
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))  # calculo del loss total

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables) # computo gradiente

  optimizer.apply_gradients(zip(gradients, trainable_variables))  # Aplico gradiente para ajustar los parametros del modelo

  return loss, total_loss

EPOCHS = 100

# Entrenamiento del encoder/decoder Image

for epoch in range(start_epoch, EPOCHS):
    start = time.time() # inicio cuenta de tiempo
    total_loss = 0 # reinicio cuenta loss

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss # computo loss de cada epoch

        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)

    if epoch % 5 == 0:
      ckpt_manager.save()

    print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))  # vacio vector

    hidden = decoder.reset_state(batch_size=1) # vacio vector

    temp_input = tf.expand_dims(load_image(image)[0], 0) # expando dimension de vector de entrada
    img_tensor_val = image_features_extract_model(temp_input) # obtengo tensor de la imagen
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val) # aplico modelo y obtengo los features

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden) # obtengo salidas de decoder

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy() 

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()  
        result.append(tokenizer.index_word[predicted_id]) ## voy formando el vector de resultado

        if tokenizer.index_word[predicted_id] == '<end>': # si llegue al fin de la oracion
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))  # obtengo vector de imagen

    fig = plt.figure(figsize=(10, 10))  # obtengo plot

    len_result = len(result)    

    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()    # muestro plot

# Compara con imagenes específicas las captions real y obtenida

# captions on the validation set
rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
result, attention_plot = evaluate(image)

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)

image_url = 'https://tensorflow.org/images/surf.jpg'
image_extension = image_url[-4:]
image_path = tf.keras.utils.get_file('image'+image_extension,
                                     origin=image_url)

result, attention_plot = evaluate(image_path)
print ('Prediction Caption:', ' '.join(result))
plot_attention(image_path, result, attention_plot)
# opening the image
Image.open(image_path)


