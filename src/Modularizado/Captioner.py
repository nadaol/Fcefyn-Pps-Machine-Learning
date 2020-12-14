## -- CAPTIONER
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
import Dataset as ds

# Path para el caching y lectura de las imagenes
image_folder = "/workspace/datasets/COCO/train2014/"    

# Path para guardar las imagenes preprocesadas con InceptionV3
prepro_images_folder = "/workspace/pickle_saves/preprocessed_IncV3_images/"  

# Path para cargar el checkpoint
checkpoint_path = "/workspace/checkpoints/image_encoder_decoder/"  

# Path para guardar la codificacion de las imagenes (features) 
encoded_image_path = '/workspace/pickle_saves/encoded_eval_images/'
#encoded_image_path = '/workspace/pickle_saves/encoded_train_images/'

# Prefijo de las imagenes
image_prefix = 'COCO_train2014_'

MAX_CKPT = 10    #max checkpoints before overwrite
CHPK_SAVE = 10   #save between epochs
EPOCHS = 60     #training epochs

BATCH_SIZE = 64
embedding_dim = 256
units = 512
vocab_size = 5001
attention_features_shape = 64
max_length=51
enc_output_units = 16384

def get_model():
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

    class CNN_Encoder(tf.keras.Model):                  # Preprocessed image Input shape (1,64,2048)
        # El encoder pasa los features a la capa FC

        def __init__(self, embedding_dim):
            super(CNN_Encoder, self).__init__()
            self.fc = tf.keras.layers.Dense(embedding_dim,activation='relu')    #Output (1,64,256)
            self.gru = tf.keras.layers.GRU(embedding_dim,return_sequences=True,recurrent_initializer='glorot_uniform')  # Capa GRU
        def call(self, x):
            x = self.fc(x)      # Fully connected layer
            x = self.gru(x)
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

    # Cargo el modelo InceptionV3 ya entrenado con el dataset de 'imagenet'
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
                                
    new_input = image_model.input # guardo la capa input
    hidden_layer = image_model.layers[-1].output  # guardo capa output

    # Obtengo el modelo de inceptionV3
    inception_model = tf.keras.Model(new_input, hidden_layer)

    encoder = CNN_Encoder(embedding_dim)    # obtengo modelo de encoder
    decoder = RNN_Decoder(embedding_dim, units, vocab_size) # obtengo modelo de decoder

    return inception_model,encoder,decoder

# Retorna la imagen image_path reducida ( shape = 299,299,3 ) y normalizada para luego utilizarla como input del inceptionV3
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img) #normilize pixels [-1 , 1]
    return img, image_path

def cache_incv3_images(images_names):
    images_names = sorted(set(images_names)) # Crea el set de imagenes no repetidas de img_name_vector y lo guarda en encode_train
    print("Number of Images for caching % d \n" % (len(images_names))) #max 82783 images
    for i,image_name in enumerate(images_names): 
        images_names[i] = image_folder + image_name + ".jpg"

    # Creo el dataset con el path de las imagenes ordenado
    image_dataset = tf.data.Dataset.from_tensor_slices(images_names)
    # Divide el dataset por batches
    image_dataset = image_dataset.map(
    load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    #Obtiene la imagen preprocesada por el inceptionV3 y la guarda en prepro_images_folder
    for img, path in image_dataset:
        batch_features = image_features_extract_model(img)#Toma las imagenes presprocesadas y las pasa por el inceptionV3
        batch_features = tf.reshape(batch_features,(batch_features.shape[0], -1, batch_features.shape[3])) # cambio dimension

    # Batch of features (BATCH_SIZE, 64, 2048)
    #Une la lista de ´'BATCH_SIZE' imagenes preprocesadas (con el inceptionV3) con el path de la imagen y la guarda en path (individalmente a cada imagen del batch)
    for bf, p in zip(batch_features, path):   #zip une varias listas en un unico diccionario
        path_of_feature = p.numpy().decode("utf-8") 
        image_id = np.char.rpartition(np.char.rpartition(path_of_feature,'_')[2],'.')[0]
        np.save(prepro_images_folder + image_prefix + image_id , bf.numpy())  #guardo batch features en un zip 

def load_checkpoint(encoder,decoder):

    optimizer = tf.keras.optimizers.Adam()

    # Creamos objeto checkpoint para el encoder y decoder ya entrenado previamente
    ckpt = tf.train.Checkpoint(encoder=encoder,decoder=decoder,optimizer = optimizer)

    # Establezco el checkpoint manager a usar (limite para 5 ultimos checkpoints)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=MAX_CKPT)

    start_epoch = 0

    if ckpt_manager.latest_checkpoint: # si existen checkpoints
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)  # cargo el ultimo checkpoint disponible  
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))

    else:
        print("Initializing from scratch.")

    return ckpt_manager,optimizer,start_epoch

def loss_function(real, pred,loss_object):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)   # Obtengo loss entre lo obtenido y lo esperado

  mask = tf.cast(mask, dtype=loss_.dtype) 
  loss_ *= mask

  return tf.reduce_mean(loss_)

@tf.function
def train_step(img_tensor,encoder,decoder, target,loss_object,optimizer,tokenizer):
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
          loss += loss_function(target[:, i], predictions,loss_object)
          # Usando teacher forcing
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))  # calculo del loss total

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables) # computo gradiente

  optimizer.apply_gradients(zip(gradients, trainable_variables))  # Aplico gradiente para ajustar los parametros del modelo

  return loss, total_loss

def train(start_epoch,encoder,decoder,dataset_train,ckpt_manager,optimizer):

    # Entrenamiento del encoder/decoder Image

    print("\n-------------  Starting %d epoch's training for captioner model  ------------\n"% (EPOCHS) )
    print("Saving model every %d epochs at %s\n"%(CHPK_SAVE,checkpoint_path))
    train_losses = []
    tokenizer = ds.load_tokenizer()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(  
    from_logits=True, reduction='none')  ## UTILIZO CROSS ENTROPY

    for epoch in range(start_epoch, EPOCHS):
        start = time.time() # inicio cuenta de tiempo
        total_loss = 0 # reinicio cuenta loss
        num_steps=0

        for (batch, (img_tensor, target)) in enumerate(dataset_train):
            batch_loss, t_loss = train_step(img_tensor,encoder,decoder,target,loss_object,optimizer,tokenizer)
            total_loss += t_loss # computo loss de cada epoch
            num_steps+=1 

            if batch % 100 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        # storing the epoch end loss value to plot later
        train_losses.append(total_loss / BATCH_SIZE)

        if ( (epoch+1) % CHPK_SAVE ) == 0:
            ckpt_manager.save(checkpoint_number=epoch)

        print ('Epoch {} Loss {:.6f}'.format(epoch + 1,total_loss/num_steps),flush=True)
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start)) 
    return train_losses


# Codifico la imagen image y la guardo en save_path .
def save_embedding(image_path,inception_model,encoder,save_path):
    
    attention_plot = np.zeros((max_length, attention_features_shape))
    image_file = image_path +  ".jpg"   # obtengo nombre el path completo de la imagen
    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image_file)[0], 0)  # expando dimension de vector de entrada
    img_tensor_val = inception_model(temp_input)  # Preprocesado por IncV3 (1,8,8,2048)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3])) #reshaped to (1,64,2048)
    features = encoder(img_tensor_val)  # aplico modelo del codificador y obtengo la codificacion de la imagen (feature) (1,64,256)
    image_name = image_path.rsplit('/',1)[1]
    # Guardo la feature de la imagen en encoded_image_path .
    with open(save_path+image_name+".emb", 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

def generate_embeddings(img_names,save_path):

    # Evaluation images encoding
    img_names_not_repeated = set(img_names)

    print("\n---------------- Starting generation of image codifications ---------------\n")
    print("\nNumber of images to encode : ",len(img_names_not_repeated))
    print("Saving in %s*.emb\n" % (save_path) )

    # Codifica las imagenes del set de evaluacion y las guarda en encoded_image_path .


    for img_name in tqdm(img_names_not_repeated):
        if (img_name[-4:]==".jpg"):           #checkea que la extensión sea jpg
            save_embedding(img_name[:-4],save_path) #codificacion y guardado

def evaluate(image_path,inception_model,encoder,decoder):
  attention_plot = np.zeros((max_length, attention_features_shape))  # vacio vector

  hidden = decoder.reset_state(batch_size=1) # vacio vector

  temp_input = tf.expand_dims(load_image(image_path)[0], 0) # Preprocesado de imagen
  img_tensor_val = inception_model(temp_input) # Preprocesado por IncV3 (1,8,8,2048)
  img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3])) #reshaped to (1,64,2048)
  print("---------- Preprocessed image shape (encoder input) %s"%img_tensor_val.shape) 
  features = encoder(img_tensor_val) # aplico modelo y obtengo los features
  print("---------- Encoder features shape (image embedding,encoder output) : %s"%features.shape) #(1,64,256)
  tokenizer = ds.load_tokenizer()
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
    plt.show() 

inception_model,encoder,decoder = get_model()
dataset_train,dataset_eval = ds.get_prepro_images_caption_datasets()
ckpt_manager,optimizer,start_epoch=load_checkpoint(encoder,decoder)
train_losses = train(start_epoch,encoder,decoder,dataset_train,ckpt_manager,optimizer)
#result,attention_plot =evaluate(image_folder+image_prefix+'000000000009.jpg',inception_model,encoder,decoder)
#print ('Prediction Caption:', ' '.join(result))