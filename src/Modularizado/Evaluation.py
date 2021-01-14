import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import unicodedata
import re
import math
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
import Dataset as ds
import Text_encoder as txt_enc

# hyperparametro k de recall
RECALL_K = 5
EVAL_LIMIT = 1000
QUERIES_LIMIT = 100

# Path para cargar las codificaciones de las imagenes 
ENCODED_IMAGES_PATH = '/workspace/pickle_saves/encoded_eval_images_lstm_30/'
#encoded_image_path = '/workspace/pickle_saves/encoded_train_images/'

# Path para cargar las codificaciones de las captions
ENCODED_CAPTIONS_PATH = '/workspace/pickle_saves/encoded_eval_captions_noconvol/'
#encoded_captions_path = '/workspace/pickle_saves/encoded_train_captions/'

enc_caption_tsv = '/workspace/Evaluations/Tsv/'
caption_metadata_tsv = '/workspace/Evaluations/Tsv/'

enc_img_tsv = '/workspace/Evaluations/Tsv/'
img_metadata_tsv = '/workspace/Evaluations/Tsv/'


def print_nearest_images(custom_caption,encoded_image_paths,encoded_images):
    caption = []
    caption.append(custom_caption)
    tokenizer = ds.load_tokenizer()
    caption = tokenizer.texts_to_sequences(caption)
    print(caption)
    caption = tf.keras.preprocessing.sequence.pad_sequences(caption, padding='post',maxlen=52)
    print(caption.shape)
    encoder = txt_enc.get_model()
    encoder,ckpt_manager,optimizer,start_epoch = txt_enc.load_checkpoint(encoder,'ckpt-2')
    encoded_caption = txt_enc.evaluate(encoder,caption)
    nearest_k,correlated_img_error = search_nearest_k_encoded_images(encoded_caption,'/None_312',encoded_image_paths,encoded_images)
    tf.print(nearest_k)

def get_encoded_captions_paths(limit):
    #Cargo las captions de evaluacion codificadas (.embt) en encoded_captions
    encoded_captions_paths=glob.glob(ENCODED_CAPTIONS_PATH + '*.emdt') 
    encoded_captions_paths = sorted(encoded_captions_paths)
    encoded_captions_paths = encoded_captions_paths[:limit]
    encoded_captions = np.zeros((len(encoded_captions_paths),16384),dtype='float32')    
    for i in range(len(encoded_captions_paths)):
        with open(encoded_captions_paths[i], 'rb') as handle:
            encoded_captions[i] = pickle.load(handle)
    return encoded_captions_paths,encoded_captions
        
#Returns list of image embeddings paths
def get_encoded_images_paths(limit):
    encoded_images_paths= glob.glob(ENCODED_IMAGES_PATH + '*.emb') # cargo todos los nombres de las codificaciones de las imagenes
    encoded_images_paths = sorted(encoded_images_paths)
    encoded_images_paths = encoded_images_paths[:limit]
    encoded_images = np.zeros((len(encoded_images_paths),16384),dtype='float32') 
    for i in range(len(encoded_images_paths)):
        with open(encoded_images_paths[i], 'rb') as handle:
            encoded_images[i] = pickle.load(handle)
    return encoded_images_paths,encoded_images

# retorno lista con los errores y los id's de las k codificaciones mas cercanas a la codificacion de imagen pasada como argumento
def search_nearest_k_encoded_captions(encoded_image,encoded_image_path,encoded_captions_paths,encoded_captions):

    nearest_k = [] 
    loss_object = tf.keras.losses.MeanSquaredError()
    # Defino error de llenado inicial
    min_error_encoded_captions = 1000000
    min_error_id = "No encontre" 

    for i in range(RECALL_K):# si la lista de los 10 errores minimos esta incompleta la lleno con min_error_encoded_captions (100000)
        nearest_k.append([min_error_encoded_captions,min_error_id])

    for i,encoded_caption in enumerate(encoded_captions):
        error = loss_object(encoded_caption,encoded_image)  # comparo textos codificados e imagen (img_tensor)
                                                 
        for k in range(len(nearest_k)):    # Si la lista nearest_k esta llena (10) comparo el error obtenido con los diez y si es menor
            if error < nearest_k[k][0]:                               # a alguno lo insterto en thebest
                encoded_caption_id = encoded_captions_paths[i].rsplit('/',1)[1].split('_')[1]
                caption_uid = int(encoded_captions_paths[i].rsplit('_',1)[1][:-5])
                nearest_k.insert(k,[error,encoded_caption_id,caption_uid])
                del nearest_k[RECALL_K]
                break

    return nearest_k

# retorno lista con los errores y los id's de las k codificaciones mas cercanas a la codificacion de imagen pasada como argumento
def search_nearest_k_encoded_images(encoded_caption,encoded_caption_path,encoded_image_paths,encoded_images):

    nearest_k = [] 
    loss_object = tf.keras.losses.MeanSquaredError()
    # Defino error de llenado inicial
    min_error_encoded_images = 1000000
    min_error_id = "No encontre" 
    correlated_img_error = 0
    # Cargo la caption para realizar la consulta
    caption_id = encoded_caption_path.rsplit('/',1)[1].split('_')[1]
    #relleno la lista con min_error
    for i in range(RECALL_K):
        nearest_k.append([min_error_encoded_images,min_error_id])

    #itero sobre todas las imagenes para calcular la distancia
    for i,encoded_image_path in enumerate(encoded_image_paths):
        error = loss_object(encoded_caption,encoded_images[i])  # comparo textos codificados e imagen (img_tensor)
        encoded_img_id = encoded_image_path.rsplit('/',1)[1].split('_')[2][:12]
        if(caption_id == encoded_img_id):
            correlated_img_error = error #same image,caption id associated error

        for k in range(RECALL_K):    # Comparo el error obtenido con los diez y si es menor
            if error < nearest_k[k][0]:                               # a alguno lo insterto en thebest
                nearest_k.insert(k,[error,encoded_img_id])
                del nearest_k[RECALL_K]
                break

    return nearest_k,correlated_img_error

def image_validQuery (nearest_k,encoded_image_path,captions_list,tokenizer):
	relevant_elements_count = 0
	image_id = encoded_image_path.rsplit('/',1)[1].split('_')[2][:12]
	print("\n\nEmbedding Image Query : %s \n"%encoded_image_path)
	for error,encoded_caption_id,caption_uid in nearest_k:
		tf.print("Retrieved caption_id [%s] , error : " %(encoded_caption_id),error)
		#cap_seq_to_string(cap_train[caption_uid-1])
		print("\n%s\n" % (ds.cap_seq_to_string(captions_list[int(caption_uid)-1],tokenizer)))
		if(encoded_caption_id == image_id): # imagen -> 1 captions  = 1
			#relevant_elements_count+=1
			relevant_elements_count=1
	return relevant_elements_count

# Returns number of relevant elements retrieved in the query 
def caption_validQuery (nearest_k,corr_img_error,encoded_caption_path,captions_list,tokenizer):
        relevant_elements_count = 0
        caption_id = encoded_caption_path.rsplit('/',1)[1].split('_')[1]
        caption_uid = encoded_caption_path.rsplit('_',1)[1][:-5]
        print("\n\nEmbedding Caption Query (%s) : %s \n"% (caption_uid,encoded_caption_path))
        print("\n%s\n" % (ds.cap_seq_to_string(captions_list[int(caption_uid)-1],tokenizer)))
        #cap_seq_to_string(cap_train[int(caption_uid)-1]) 
        for error,encoded_image_id in nearest_k:
                tf.print("Retrieved image id [%s] error "% (encoded_image_id),error )
                if(caption_id == encoded_image_id): # imagen -> 1 captions  = 1
                        relevant_elements_count+=1
        tf.print("Correlated Image error :  ",corr_img_error)
        return relevant_elements_count

def get_image_caption_Recall(encoded_captions_paths,encoded_images_paths,encoded_captions,encoded_images):
	print("-------------- Calculating recall@%d for a %d size image set and %d size captions set  -------------- \n\n" % (RECALL_K,len(encoded_images_paths),len(encoded_captions_paths)))
	print('Reading encoded captions from %s \n' % (ENCODED_CAPTIONS_PATH))
	print('Reading encoded images from %s \n' % (ENCODED_IMAGES_PATH))
	relevant_captions_count = 0
	images_names_train, images_names_eval , captions_train, captions_eval , tokenizer = ds.get_image_names_captions_lists()
	for i,encoded_image_path in enumerate(encoded_images_paths):
		nearest_k = search_nearest_k_encoded_captions(encoded_images[i],encoded_image_path,encoded_captions_paths,encoded_captions)
		relevant_captions_count+=image_validQuery(nearest_k,encoded_image_path,captions_eval,tokenizer)
		i+=1
		if(i == QUERIES_LIMIT):
			break
		if(i % 10 == 0):
			print("\n----- Queries : %d -----\n"%i)
	#print("\n\nRelevant captions retrieved : %d , Total relevant captions %d \n "%(relevant_captions_count,i*5))
	print("\n\nValid queries : %d , Total queries %d \n "%(relevant_captions_count,i))
    #return (relevant_captions_count/(i*5))
	return (relevant_captions_count/(i))

def get_caption_image_Recall(encoded_captions_paths,encoded_images_paths,encoded_captions,encoded_images):
		print("-------------- Calculating recall@%d for a %d size caption set -> %d size image set -------------- \n\n" % (RECALL_K,len(encoded_captions_paths),len(encoded_images_paths)))
		print('Reading encoded captions from %s \n' % (ENCODED_CAPTIONS_PATH))
		print('Reading encoded images from %s \n' % (ENCODED_IMAGES_PATH))
		relevant_images_count = 0
		images_names_train, images_names_eval , captions_train, captions_eval , tokenizer = ds.get_image_names_captions_lists()
		for i,encoded_caption_path in enumerate(encoded_captions_paths):
			nearest_k,corr_img_error = search_nearest_k_encoded_images(encoded_captions[i],encoded_caption_path,encoded_images_paths,encoded_images)
			relevant_images_count+=caption_validQuery(nearest_k,corr_img_error,encoded_caption_path,captions_eval,tokenizer)
			if(i == QUERIES_LIMIT):
				break
			if(i % 10 == 0):
				print("\n---- Queries : %d -----\n"%i)
		print("\n\nRelevant elements retrived in all quieries : %d , Total Relevant elements in all queries : %d \n " % (relevant_images_count,i))
		return (relevant_images_count/(i))
    

# Recall@10 = query(1 elemento de los 10 encontrados es valido)/total_queries
# Recall@k =  total de elementos correctos devueltos de todas las quieries/ total de elementos correctos esperados
# elementos esperados por cada query = 5 (captions por imagenes correctas)


def map_captions(caption_seq,cap_path,cap_tensor):
    tokenizer = ds.load_tokenizer()
    img_id = cap_path.split('/')[-1].split('_')[1]
    caption_string = ds.cap_seq_to_string(caption_seq,tokenizer)
    return cap_tensor,caption_string,img_id

#COCO_train2014_000000465301
def map_images(img_path,img_tensor):
    img_id = img_path.split('/')[-1].split('_')[2][:12]

    return img_tensor,'None',img_id

def get_embedding_lists(limit):

    images_names_train, images_names_eval , captions_train, captions_eval , tokenizer = ds.get_image_names_captions_lists()
    encoded_images_paths,encoded_images = get_encoded_images_paths(limit)
    encoded_captions_paths,encoded_captions = get_encoded_captions_paths(limit)

    captions_list = []
    images_list = []

    for i,cap_path in enumerate(encoded_captions_paths):
        captions_list.append(map_captions(captions_eval[i],cap_path,encoded_captions[i]))

    img_path_aux = ''
    i=0
    for image_path,img_tensor in zip(encoded_images_paths,encoded_images):
        if(img_path_aux != image_path):
            images_list.append(map_images(image_path,img_tensor))
            i+=1
        else :
            img_path_aux = image_path

        if i==(math.floor(limit/5)):
            break

    return captions_list,images_list

def write_embeddings(caption_list,image_list,file_name,metadata1_title,metadata2_title,tensors_save_path,metadata_save_path):

    tensors_file = open(tensors_save_path+file_name+'_tensor.tsv', "w")
    metadata_file = open(metadata_save_path+file_name+'_metadata.tsv', "w")

    metadata_file.write(metadata1_title + '\t' + metadata2_title+'\n')

    embedding_list = caption_list + image_list

    for tensor,metadata1,metadata2 in embedding_list:
        metadata_file.write(" ".join(metadata1)+'\t'+metadata2)
        for tensor_element in tensor:
            tensors_file.write(str(tensor_element)+'\t')
        tensors_file.write('\n')
        metadata_file.write('\n')
    print('Writed tsv file with %d captions and %d images \n' % (len(caption_list),len(image_list)) )

caption_list,image_list = get_embedding_lists(1000)

#write_embeddings(caption_list,image_list,'Captions','Caption','Image_Id',enc_caption_tsv,caption_metadata_tsv)


""" 
encoded_captions_paths,encoded_captions = get_encoded_captions_paths(EVAL_LIMIT)
encoded_images_paths,encoded_images = get_encoded_images_paths(EVAL_LIMIT/5)
get_image_caption_Recall(encoded_captions_paths,encoded_images_paths,encoded_captions,encoded_images)
#get_caption_image_Recall(encoded_captions_paths,encoded_images_paths,encoded_captions,encoded_images)
#print_nearest_images('<start> a group of pictures including food and beverages <end>',encoded_images_paths,encoded_images) 
"""