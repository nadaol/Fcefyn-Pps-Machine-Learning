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
import glob


## ----- PATHS variables for docker execution (mapped volume : workspace )

# Path para la lectura de las annotations
annotation_folder = '/workspace/datasets/COCO/annotations'
annotation_file = annotation_folder + '/captions_train2014.json'

# Path para cargar el checkpoint del encoder y evaluar
checkpoint_path = '/workspace/checkpoints/text_encoder/'

# Path para cargar el tokenizer
pickle_tokenizer_path = '/workspace/pickle_saves/tokenizer/tokenizer.pickle'

# Path para guardar la salida del codificador
#encoded_captions_path = '/workspace/pickle_saves/encoded_captions_eval/'
encoded_captions_path  = '/workspace/pickle_saves/encoded_captions/'

# Path cargar embedding imagenes del set de evaluacion
#encoded_image_path = '/workspace/pickle_saves/encoded_images_eval/'
# Set de entrenamiento
encoded_image_path = '/workspace/pickle_saves/encoded_images/'

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


# Aplico padding a las captions , para obtener captions con tamaño fijo = max_length
all_all_captions = tf.keras.preprocessing.sequence.pad_sequences(all_all_captions,padding='post')

#all_captions [0] = [  3   2 136 491  10 622 430 271  58   4   0   0  ....  0   0   0   0   0   0   0   0   0   0   0   0]

#caption int array to caption sentence
def cap_seq_to_string(caption_seq):
  for word_number in caption_seq:
    print("%s " % tokenizer.index_word[word_number],end='',flush=True)
  print("\n\n")

#Split train,val dataset
TRAIN_PERCENTAGE = 0.8
train_examples = int (TRAIN_PERCENTAGE*num_examples)
all_img_name_vector, img_name_val , all_captions, cap_val = all_all_img_name_vector[:train_examples] , all_all_img_name_vector[train_examples:] , all_all_captions[:train_examples] ,all_all_captions[train_examples:]

# Mezclado de captions e imagenes (random_state 1) train y evaluacion
#train set
cap_train,img_name_train = shuffle(all_captions,all_img_name_vector,random_state=1) 
#eval set
cap_val, img_name_val = shuffle(cap_val,img_name_val,random_state=1) 



# Comparar todas las codificaciones de las captions del set de evaluacion con todas las imagenes codificadas.
# Encontrar para cada codificacion de texto las k codificaciones de las imagenes mas cercanas (distancia euclidiana)
# Si dentro de esas k codificaciones de las imagenes se encuentra la caption buscada (mismo image_id) el elemento es el buscado.
# Se divide el total de elementos encontrados por el total de consultas para sacar el recall@k .

# hyperparametro k de recall
RECALL_K = 5
EVAL_LIMIT = 1000
QUERIES_LIMIT = 100

# Obtengo objeto de loss
loss_object = tf.keras.losses.MeanSquaredError()

#Cargo las captions de evaluacion codificadas (.embt) en encoded_captions
encoded_captions_paths=glob.glob(encoded_captions_path + '*.emdt') 
encoded_captions_paths = sorted(encoded_captions_paths)
encoded_captions_paths = encoded_captions_paths[:EVAL_LIMIT*5]
encoded_captions = np.zeros((len(encoded_captions_paths),16384),dtype='float32')
# (320,16384)
for i in range(len(encoded_captions_paths)):
    with open(encoded_captions_paths[i], 'rb') as handle:
        encoded_captions[i] = pickle.load(handle) 

#Cargo las captions de evaluacion codificadas (.embt) en encoded_captions
encoded_image_paths= glob.glob(encoded_image_path + '*.emb') # cargo todos los nombres de las codificaciones de las imagenes
encoded_image_paths = sorted(encoded_image_paths)
encoded_image_paths = encoded_image_paths[:EVAL_LIMIT]

# retorno lista con los errores y los id's de las k codificaciones mas cercanas a la codificacion de imagen pasada como argumento
def search_nearest_k_encoded_images(caption_embedding_name_path):

    nearest_k = [] 
    # Defino error de llenado inicial
    min_error_encoded_images = 1000000
    min_error_id = "No encontre" 
    # Cargo la caption para realizar la consulta
    caption_tensor = np.load(caption_embedding_name_path,allow_pickle = True)
    caption_id = caption_embedding_name_path.rsplit('/',1)[1].split('_')[1]

    #relleno la lista con min_error
    for i in range(RECALL_K):
        nearest_k.append([min_error_encoded_images,min_error_id])

    #itero sobre todas las imagenes para calcular la distancia
    for i,encoded_image_path in enumerate(encoded_image_paths):
        img_tensor = np.load(encoded_image_path,allow_pickle = True)
        error = loss_object(caption_tensor,img_tensor)  # comparo textos codificados e imagen (img_tensor)
        encoded_img_id = encoded_image_path.rsplit('/',1)[1].split('_')[2][:12]
        #if(caption_id == encoded_img_id):
            #print("Image %s  error : %f \n" % (encoded_img_id,error) )

        for k in range(RECALL_K):    # Si la lista nearest_k esta llena (10) comparo el error obtenido con los diez y si es menor
            if error < nearest_k[k][0]:                               # a alguno lo insterto en thebest
                nearest_k.insert(k,[error,encoded_img_id])
                del nearest_k[RECALL_K]
                break

    return nearest_k

# Returns number of relevant elements retrieved in the query 
def validQuery (nearest_k,caption_embedding_path):
        relevant_elements_count = 0
        caption_id = caption_embedding_path.rsplit('/',1)[1].split('_')[1]
        caption_uid = caption_embedding_path.rsplit('_',1)[1][:-5]
        print("\n\nEmbedding Caption Query (%s) : %s \n"% (caption_uid,caption_embedding_path))
        cap_seq_to_string(cap_train[int(caption_uid)-1]) 
        for error,encoded_image_id in nearest_k:
                tf.print("Retrieved image id [%s] error "% (encoded_image_id),error )
                if(caption_id == encoded_image_id): # imagen -> 1 captions  = 1
                        relevant_elements_count+=1
        return relevant_elements_count

#print("valid queries %d , total queries : 1 , image_id : %d"% (valid_queries,i))
#print(validQuery(search_nearest_k_encoded_captions(encoded_image_paths[0]),encoded_image_paths[0]))

# Iterate over all captions to query relevant images and calculate recall
def get_Recall():
        print("-------------- Calculating recall@%d for a %d size caption set -> %d size image set -------------- \n\n" % (RECALL_K,len(encoded_captions_paths),len(encoded_image_paths)))
        #print(encoded_image_paths)
        #print(encoded_captions_paths)
        relevant_images_count = 0
        for i,encoded_caption_path in enumerate(encoded_captions_paths):
                relevant_images_count+=validQuery(search_nearest_k_encoded_images(encoded_caption_path),encoded_caption_path)
                i+=1
                if(i == QUERIES_LIMIT):
                        break
                if(i % 10 == 0):
                        print("\n---- Queries : %d -----\n"%i)
        print("\n\nRelevant elements retrived in all quieries : %d , Total Relevant elements in all queries : %d \n "%(relevant_images_count,i))
        return (relevant_images_count/(i))

print("\nRecall@%d : %f"%(RECALL_K,get_Recall()*100))
