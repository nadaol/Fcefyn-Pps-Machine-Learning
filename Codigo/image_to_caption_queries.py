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







## ----- Dataset creation

# Path para la lectura de las annotations
annotation_folder = '/workspace/datasets/COCO/annotations'
annotation_file = annotation_folder + '/captions_train2014.json'

# Path para cargar el checkpoint del encoder y evaluar
checkpoint_path = '/workspace/checkpoints/text_encoder/'

# Path para cargar el tokenizer
pickle_tokenizer_path = '/workspace/pickle_saves/tokenizer/tokenizer.pickle'

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
all_captions, all_img_name_vector = shuffle(all_captions,all_img_name_vector,random_state=1) 
#eval set
cap_val, img_name_val = shuffle(cap_val,img_name_val,random_state=1) 



# Comparar todas las codificaciones de imagenes del set de evaluacion con todas las captions codificadas.
# Encontrar para cada codificacion de imagen las k codificaciones de las captions mas cercanas (distancia euclidiana)
# Si dentro de esas k codificaciones de las captions se encuentra una caption buscada (mismo image_id) ,ese elemento es correcto.
# Se divide el total de elementos validos encontrados por el total de consultas * 5 (captions correctas por imagen) para sacar el recall@k .


# hyperparametro k de recall
RECALL_K = 5
EVAL_LIMIT = 32

# Path para guardar la codificacion de las imagenes (features) 
encoded_image_path = '/workspace/pickle_saves/encoded_images_eval/'

# Path para guardar la salida del codificador
encoded_captions_path = '/workspace/pickle_saves/encoded_captions_eval/'
 
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
                caption_uid = int(encoded_captions_paths[i].rsplit('_',1)[1][:-5])
                nearest_k.insert(k,[error,encoded_caption_id,caption_uid])
                del nearest_k[RECALL_K]
                break

    return nearest_k


def validQuery (nearest_k,image_embedding_path):
	relevant_elements_count = 0
	image_id = image_embedding_path.rsplit('/',1)[1].split('_')[2][:12]
	print("\n\nEmbedding Image Query : %s \n"%image_embedding_path)
	for error,encoded_caption_id,caption_uid in nearest_k:
		tf.print("Retrieved caption_id [%s] , error : " %(encoded_caption_id),error)
		cap_seq_to_string(cap_val[caption_uid])
		if(encoded_caption_id == image_id): # imagen -> 1 captions  = 1
			relevant_elements_count+=1
            #print("caption id %s \n"%encoded_caption_id)
	return relevant_elements_count

#print("valid queries %d , total queries : 1 , image_id : %d"% (valid_queries,i))
#print(validQuery(search_nearest_k_encoded_captions(encoded_image_paths[0]),encoded_image_paths[0]))

def get_Recall():
	print("-------------- Calculating recall@%d for a %d size image set and %d size captions set  -------------- \n\n" % (RECALL_K,len(encoded_image_paths),len(encoded_captions_paths)))
	print(encoded_image_paths)
	print(encoded_captions_paths)
	relevant_captions_count = 0
	for i,encoded_image_path in enumerate(encoded_image_paths):
		relevant_captions_count+=validQuery(search_nearest_k_encoded_captions(encoded_image_path),encoded_image_path)
		i+=1
		if(i % 10 == 0):
			print("\n----- Queries : %d -----\n"%i)
	print("\n\nRelevant captions retrieved : %d , Total relevant captions %d \n "%(relevant_captions_count,i*5))
	return (relevant_captions_count/(i*5))

print("\nRecall@%d : %f"%(RECALL_K,get_Recall()*100))
    

# Recall@10 = query(1 elemento de los 10 encontrados es valido)/total_queries
# Recall@k =  total de elementos correctos devueltos de todas las quieries/ total de elementos correctos esperados
# elementos esperados por cada query = 5 (captions por imagenes correctas)
