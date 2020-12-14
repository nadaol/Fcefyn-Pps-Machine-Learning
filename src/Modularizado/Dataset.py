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

# Path para la lectura de las annotations
annotation_folder = '/workspace/datasets/COCO/annotations/'
annotation_file = annotation_folder + 'captions_train2014.json'

# Path para la lectura de las imagenes
image_folder = "/workspace/datasets/COCO/train2014/"  

# Path para cargar las imagenes preprocesadas con InceptionV3
prepro_images_folder = "/workspace/pickle_saves/preprocessed_IncV3_images/"  

# Path para cargar/guardar las codificaciones de imagenes del set de entrenamiento
encoded_image_path = '/workspace/pickle_saves/encoded_train_images/'
# Path para cargar/guardar las codificaciones de imagenes del set de evaluacion
encoded_image_path_eval = '/workspace/pickle_saves/encoded_eval_images/'

# Prefijo de las imagenes
image_prefix = 'COCO_train2014_'

# Path para guardar el tokenizer
pickle_tokenizer_path = '/workspace/pickle_saves/tokenizer/tokenizer.pickle'

BATCH_SIZE = 64             #Sizes must be multiple of batch size
BUFFER_SIZE = 1000
TRAIN_PERCENTAGE = 0.8      #Split between train and evaluation data
MAX_LEN = 414113         #Limit of total captions/images before split , max 414113
VOCAB_LIMIT = 5000          # Limite del vocabulario a k palabras.

def download_dataset():
    # ------------------------------  Descargo captions e imagenes,si no existen
    if not os.path.exists(annotation_file):
        annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                cache_subdir=os.path.abspath('.'),
                                                origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                extract = True)
        os.remove(annotation_zip)

    if not os.path.exists(image_folder):
        image_zip = tf.keras.utils.get_file('train2014.zip',
                                            cache_subdir=os.path.abspath('.'),
                                            origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                            extract = True)
        os.remove(image_zip)

def cap_seq_to_string(caption_seq,tokenizer):
    sentence = []
    for word_number in caption_seq:
        sentence.append(tokenizer.index_word[word_number])
    return sentence

def get_image_names_captions_lists():

    download_dataset()

    # ---------------------------------  Lectura de annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    #Sort annotations by image_id
    annotations['annotations'] = sorted(annotations['annotations'], key = lambda i: i['image_id']) 

    ## Cargado de captions y path de las imagenes correspondientes
    captions = []
    images_paths = []
    train_examples = int(TRAIN_PERCENTAGE*MAX_LEN)

    for annot in annotations['annotations']:  #not in order
        caption = '<start> ' + annot['caption'] + ' <end>' # Parseo annotations agregando simbolos de inicio y fin .
        image_id = annot['image_id']                        # obtengo id de la imagen correspondiente al caption

        full_coco_image_path = image_prefix + '%012d' % (image_id) # guardo el path a las codificaciones de las imagenes (.emb)

        images_paths.append(full_coco_image_path)  # Guardo el path imagen
        captions.append(caption)                      # Guardo respectivo caption                     

    # Limitar a MAX_LEN captions-imagenes (414113 captions en total)(82783 images) para luego usar en el entrenamiento
    #  num_images ~ MAX_LEN/5
    captions = captions[:MAX_LEN]   # string train captions
    images_paths = images_paths[:MAX_LEN] # 

    # Obtenemos tokenizer para dividir las captions en palabras
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_LIMIT,
                                                    oov_token="<unk>",
                                                    filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(captions)

    tokenizer.word_index['<pad>'] = 0     # defino valor para pad
    tokenizer.index_word[0] = '<pad>'

    # Crear vectores tokenizados
    captions = tokenizer.texts_to_sequences(captions) # obtengo secuencia de enteros que matchean las captions (MAX_LEN captions)...segunda vez

    # Aplico padding a las captions 
    captions = tf.keras.preprocessing.sequence.pad_sequences(captions, padding='post')

    # Separacion de conjuntos de entrenamiento y de evaluacion (al crear los datasets el ultimo batch si es menor a BATCH_SIZE se elimina)
    images_names_train, images_names_eval , captions_train, captions_eval = images_paths[:train_examples] , images_paths[train_examples:] , captions[:train_examples] , captions[train_examples:]

    train_rest = (train_examples%BATCH_SIZE)
    eval_rest = ((MAX_LEN-train_examples)%BATCH_SIZE)
    print("Limit of examples : %d\nTrain Split [%d - %d] \nCaption Split [%d - %d]"%(MAX_LEN,0,train_examples-train_rest,train_examples,MAX_LEN-eval_rest))

    return images_names_train, images_names_eval , captions_train, captions_eval , tokenizer

# Funcion para mapear nomre de la imagen a tensor preprocesado por inceptionV3 y la caption correspondiente
def map_func_prepro(img_name, cap):
    img_name_decoded = img_name.decode('utf-8')+'.npy'
    path = prepro_images_folder + img_name_decoded
    prepro_image = np.load(path)
    return prepro_image,cap

def load_tokenizer():
    with open(pickle_tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

#caption int array to caption sentence
def cap_seq_to_string(caption_seq,tokenizer):
    sentence = []
    for word_number in caption_seq:
        sentence.append(tokenizer.index_word[word_number])
    return sentence

def get_prepro_images_caption_datasets():

    images_names_train, images_names_eval , captions_train, captions_eval , tokenizer = get_image_names_captions_lists()

    print("img train len (last batch is dropped) : %d\n"% (len(images_names_train)) )
    print("img eval len (last batch is dropped) : %d\n"% (len(images_names_eval)) )

    # Creo ambos datasets con los nombres de imagenes y la captions correspondientes,mezclo.
    dataset_train = tf.data.Dataset.from_tensor_slices((images_names_train,captions_train)).shuffle(BUFFER_SIZE,reshuffle_each_iteration=True)
    dataset_eval = tf.data.Dataset.from_tensor_slices((images_names_eval,captions_eval)).shuffle(BUFFER_SIZE,reshuffle_each_iteration=True)

    #Impresion primeros datos de los datasets creados
    eval_images_name,eval_caption= next(iter(dataset_eval))
    train_images_name,train_caption = next(iter(dataset_train))
    print("----- First train dataset input images (before map)  : %s \n----- Correlated caption \n %s\n\n" % (train_images_name,cap_seq_to_string(np.array(train_caption),tokenizer)))
    print("----- First eval dataset input images (before map) : %s \n----- Correlated caption \n %s\n\n" % (eval_images_name,cap_seq_to_string(np.array(eval_caption),tokenizer)))

    # Mapeo nombres de imagenes con los tensores preprocesados ya guardados y separo en batches
    dataset_train = dataset_train.map(lambda item1, item2: tf.numpy_function(map_func_prepro, [item1, item2], [tf.float32, tf.int32]),num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE,drop_remainder=True)

    dataset_eval = dataset_eval.map(lambda item1, item2: tf.numpy_function(map_func_prepro, [item1, item2], [tf.float32, tf.int32]),num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE,drop_remainder=True)

    #Realizo un prefetch para mayor rendimiento de entrenamiento
    dataset_eval = dataset_eval.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset_train = dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) 

    return dataset_train,dataset_eval

def map_func_enc(img_name, cap):
    img_name_decoded = img_name.decode('utf-8')+'.emb'
    enc_img_path = encoded_image_path + img_name_decoded
    img_tensor = np.load(enc_img_path,allow_pickle=True)
    return img_tensor, cap

    #Mapea path de la imagen con la codificacion correspondiente
def map_func_enc_eval(img_name, cap):
    img_name_decoded = img_name.decode('utf-8')+'.emb'
    enc_img_path = encoded_image_path_eval + img_name_decoded
    img_tensor = np.load(enc_img_path,allow_pickle=True)
    return img_tensor, cap

def get_enc_image_caption_datasets():

    images_names_train, images_names_eval , captions_train, captions_eval , tokenizer = get_image_names_captions_lists()
    #Mapea path de la imagen con la codificacion correspondiente

    # Creo el dataset con el path de las imagenes y los captions 
    dataset_train = tf.data.Dataset.from_tensor_slices((images_names_train,captions_train))
    dataset_train = dataset_train.shuffle(BUFFER_SIZE,reshuffle_each_iteration=True)

    dataset_eval = tf.data.Dataset.from_tensor_slices((images_names_eval,captions_eval))
    dataset_eval = dataset_eval.shuffle(BUFFER_SIZE,reshuffle_each_iteration=True) 

    # Mapear dataset con las imagenes codificadas y las captions 
    dataset_train = dataset_train.map(lambda item1, item2: tf.numpy_function(
            map_func_enc, [item1, item2], [tf.float32, tf.int32]))
    dataset_eval = dataset_eval.map(lambda item1, item2: tf.numpy_function(
            map_func_enc_eval, [item1, item2], [tf.float32, tf.int32]))
    
    # Mezcla el dataset y lo divide en batches de tamano 'BATCH_SIZE' .
    dataset_train = dataset_train.batch(BATCH_SIZE,drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)     
    dataset_eval = dataset_eval.batch(BATCH_SIZE,drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE) 

    encoded_image_batch,captions_batch = next(iter(dataset_train))
    print("First input caption (train) : %s \nCorrelated encoded image %s\n\n" % (cap_seq_to_string(np.array(captions_batch),tokenizer),encoded_image_batch))

    encoded_image_batch,captions_batch = next(iter(dataset_eval))
    print("First input caption (eval) : %s \nCorrelated encoded image %s\n\n" % (cap_seq_to_string(np.array(captions_batch),tokenizer),encoded_image_batch))

    return dataset_train,dataset_eval