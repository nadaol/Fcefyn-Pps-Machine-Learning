import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

# Scikit-learn includes many helpful utilities

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
# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

# Download caption annotation files

annotation_folder = '/annotations/'
max_length_set = 49

annotation_file = os.path.abspath('.') +'/annotations/captions_train2014.json'
# Download image files
image_folder = '/img_embeddings/'
PATH = os.path.abspath('.') + image_folder
print('FIN1')
# Read the json file
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Store captions and image names in vectors
all_all_captions = []
all_all_img_name_vector = []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    all_all_captions.append(caption)
    all_all_img_name_vector.append(annot['image_id'])


#get tokenizer
# Choose the top 5000 words from the vocabulary
top_k = 5000

with open('./tokenizer_new/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Create the tokenized vectors
all_all_captions = tokenizer.texts_to_sequences(all_all_captions)
all_captions = []
all_img_name_vector  = []
for i in range(len(all_all_captions)):
    if len(all_all_captions[i]) <= max_length_set:
        all_captions.append(all_all_captions[i])
        #print(all_all_img_name_vector[i])

        all_img_name_vector.append(all_all_img_name_vector[i])

max_length = max(len(t) for t in all_captions)

all_captions = tf.keras.preprocessing.sequence.pad_sequences(all_captions,maxlen=max_length, padding='post')

# Calculates the max_length, which is used to store the attention weights
max_length = 49
BUFFER_SIZE = 1000
BATCH_SIZE = 64
#steps_per_epoch = len(img_name_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
#units = 512
output_units = 512
output_size = 16384
vocab_inp_size = top_k + 1 #ojooo


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, enc_output_units,batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.enc_output_units = enc_output_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    #self.gru = tf.keras.layers.GRU(self.enc_output_units,
    #                              return_sequences=False,
    #                               return_state=True,
    #                               recurrent_initializer='glorot_uniform')
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                  return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.cnn1 = tf.keras.layers.Conv1D(256, 4, activation='relu',input_shape = [49, 1024])
    self.fc = tf.keras.layers.Dense(enc_output_units)
    
    #fin agregado por aldo

  def call(self, x, hidden):
    x = self.embedding(x)
    #print(x)
    output_gru, state = self.gru(x, initial_state = hidden)

    #print ('output_gru',output_gru)
    #print ('state',state)
    cnn1out = self.cnn1(output_gru)
    #print ('cnn1out',cnn1out)
    flat_vec = tf.reshape(cnn1out,[cnn1out.shape[0],cnn1out.shape[1]*cnn1out.shape[2]])


    output = self.fc(flat_vec)
    #print('output',output)
    #sys.exit()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    #output = self.fc(output_gru)
    output = tf.nn.relu(output) #ver si reduce el error
    #print('output relu',output)
    #sys.exit()
    return output, state
    #return flat_vec, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))


print("parametros encoder",vocab_inp_size, embedding_dim, units, output_size, BATCH_SIZE)
#sys.exit(0)

encoder = Encoder(vocab_inp_size, embedding_dim, units, output_size, BATCH_SIZE)



#optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
optimizer = tf.keras.optimizers.Adam()
#loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#    from_logits=True, reduction='none')

loss_object = tf.keras.losses.MeanSquaredError()

def loss_function(real, pred):
  #mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  #mask = tf.cast(mask, dtype=loss_.dtype)
  #loss_ *= mask
  return tf.reduce_mean(loss_)



                                 
#checkpoint_path = './training_emb_checkpoints_pru'
checkpoint_path = './ckk_pru'
#print(tf.train.list_variables(tf.train.latest_checkpoint(checkpoint_path)))
#sys.exit()
ckpt = tf.train.Checkpoint(encoder=encoder,
                           optimizer = optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

#start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

#if ckpt_manager.latest_checkpoint:
#  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
print( "restoring the latest checkpoint in checkpoint_path")
ckpt.restore(ckpt_manager.latest_checkpoint)                           
#max_poll1D = tf.keras.layers.MaxPool1D(pool_size=2, strides=None, padding='valid')
#sys.exit()
def evaluate(cap):
  
  enc_hidden = encoder.initialize_hidden_state()
  enc_output, enc_hidden = encoder(cap, enc_hidden)
  return enc_output

annotation_folder = '/annotations/'
annotation_file = os.path.abspath('.') +'/annotations/captions_train2014.json'
# Store captions and image names in vectors
#all_captions = []
dataset = tf.data.Dataset.from_tensor_slices((all_captions,all_img_name_vector))
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
text_id = 0
for (batch,(cap,img_name)) in enumerate(dataset):
    
    #full_coco_text_path = PATH + 'encodedText_%012d_%012d.emdt' % (img_name,text_id)
    #if text_id % 200 == 0 or text_id < 10: 
    #    print (id,full_coco_text_path)
    text_encoded_vec = evaluate(cap)
    for i in range(64):
      text_id += 1
      #print(img_name[i],max(text_encoded_vec[i].numpy()))
      full_coco_text_path = PATH + 'encodedText_%012d_%012d.emdt' % (img_name[i],text_id)
      with open(full_coco_text_path, 'wb') as handle:
        pickle.dump(text_encoded_vec[i].numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)
    if batch % 20==0:
      print("batch",batch)




