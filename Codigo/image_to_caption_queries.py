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
import glob

# Comparar todas las codificaciones de imagenes del set de evaluacion con todas las captions codificadas.
# Encontrar para cada codificacion de imagen las k codificaciones de las captions mas cercanas (distancia euclidiana)
# Si dentro de esas k codificaciones de las captions se encuentra la caption buscada (mismo image_id) la consulta es valida.
# Se divide el total de consultas validas por el total de consultas para sacar el recall@k .

# hyperparametro k de recall
RECALL_K = 5
EVAL_LIMIT = 128

# Path para guardar la codificacion de las imagenes (features) 
encoded_image_path = '/workspace/pickle_saves/encoded_images_eval/'

# Path para guardar la salida del codificador
encoded_captions_path = '/workspace/pickle_saves/encoded_captions_eval/'
 
# Obtengo objeto de loss
loss_object = tf.keras.losses.MeanSquaredError()

#Cargo las captions de evaluacion codificadas (.embt) en encoded_captions
encoded_captions_paths=glob.glob(encoded_captions_path + '*.emdt') 
encoded_captions_paths = sorted(encoded_captions_paths)
encoded_captions_paths = encoded_captions_paths[:EVAL_LIMIT]

encoded_captions = np.zeros((len(encoded_captions_paths),64,256),dtype='float32')
for i in range(len(encoded_captions_paths)):
    with open(encoded_captions_paths[i], 'rb') as handle:
        encoded_captions[i] = pickle.load(handle) 

#Cargo las captions de evaluacion codificadas (.embt) en encoded_captions
encoded_image_paths= glob.glob(encoded_image_path + '*.emb') # cargo todos los nombres de las codificaciones de las imagenes
encoded_image_paths = sorted(encoded_image_paths)
encoded_image_paths = encoded_image_paths[:EVAL_LIMIT]

# retorno lista con los errores y los id's de las k codificaciones mas cercanas a la codificacion de imagen pasada como argumento
def search_nearest_k_encoded_captions(image_embedding_name_path):

    nearest_k = [] 
    # Defino error de llenado inicial
    min_error_encoded_captions = 1000000
    min_error_id = "No encontre" 

    img_tensor = np.load(image_embedding_name_path,allow_pickle = True)

    for i in range(RECALL_K):# si la lista de los 10 errores minimos esta incompleta la lleno con min_error_encoded_captions (100000)
        nearest_k.append([min_error_encoded_captions,min_error_id])

    for i,encoded_caption in enumerate(encoded_captions):
        error = loss_object(encoded_caption,img_tensor)  # comparo textos codificados e imagen (img_tensor)
                                                 
        for k in range(len(nearest_k)):    # Si la lista nearest_k esta llena (10) comparo el error obtenido con los diez y si es menor
            if error < nearest_k[k][0]:                               # a alguno lo insterto en thebest
                encoded_caption_id = encoded_captions_paths[i].rsplit('/',1)[1].split('_')[1]
                nearest_k.insert(k,[error,encoded_caption_id])
                del nearest_k[RECALL_K]
                break

    return nearest_k


def validQuery (nearest_k,image_embedding_path):
        
        image_id = image_embedding_path.rsplit('/',1)[1].split('_')[2][:12]
        #print(nearest_k)
        #print("image id %s \n"%image_id)
        for error,encoded_caption_id in nearest_k:
            #print(encoded_caption_id)
            if(encoded_caption_id == image_id): # imagen -> 1 captions  = 1
                return True
            #print("caption id %s \n"%encoded_caption_id)
        return False

#print("valid queries %d , total queries : 1 , image_id : %d"%(valid_queries,i))
#print(validQuery(search_nearest_k_encoded_captions(encoded_image_paths[0]),encoded_image_paths[0]))

def get_Recall():
    valid_queries = 0
    for i,encoded_image_path in enumerate(encoded_image_paths):
        if(validQuery(search_nearest_k_encoded_captions(encoded_image_path),encoded_image_path)):
            valid_queries = valid_queries + 1
        if(i % 10 == 0):
            print("queries : %d"%i)
    i = i +1
    print("valid queries : %d , total caption queries %d \n "%(valid_queries,i))
    return (valid_queries/(i))

print("\nRecall : %f"%(get_Recall()*100))
    
