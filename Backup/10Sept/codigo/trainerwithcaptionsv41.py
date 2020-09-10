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


annotation_file = os.path.abspath('.') +'/annotations/captions_train2014.json'
# Download image files
image_folder = '/img_embeddings/'
PATH = os.path.abspath('.') + image_folder
print('FIN1')
# Read the json file
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Store captions and image names in vectors
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = PATH + 'COCO_train2014_' + '%012d' % (image_id)

    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)

# Shuffle captions and image_names together
# Set a random state
train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)

# Select the first 30000 captions from the shuffled set
num_examples = 80000
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]
print (len(train_captions), len(all_captions))
print("FIN2")
#get tokenizer
# Choose the top 5000 words from the vocabulary
top_k = 5000

with open('./tokenizer_new/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)

# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically

cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

#prepare image batch
# Feel free to change batch_size according to your system configuration
# Get unique images
#print(len(img_name_vector)) #80000

#print(len(encode_train)) #54363
#sys.exit()
""" def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path """

def load_image(image_path):
    img = tf.io.read_file(image_path)
    return img, image_path
    
#encode_train = sorted(set(img_name_vector))
#print(type(encode_train[0]))
#sys.exit()
#image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
#image_dataset = image_dataset.map(
#  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

#for img, path in image_dataset:
  #batch_features = image_features_extract_model(img)
  #batch_features = tf.reshape(batch_features,
  #                            (batch_features.shape[0], -1, batch_features.shape[3]))
  #for bf, p in zip(batch_features, path):
#  for bf, p in zip(img, path):
    #print("SHAPE ",bf.shape) (64,2048)
  #    sys.exit
#    path_of_feature = p.numpy().decode("utf-8")
#    np.save(path_of_feature, bf.numpy())

# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)

# Create training and validation sets using an 80-20 split
#img_name_train, img_name_val, input_tensor_train,input_tensor_val = train_test_split(img_name_vector,
#                                                                    cap_vector,
#                                                                    test_size=0.2,
#                                                                    random_state=0)
# Create training and validation sets using an 80-20 split
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)

#print(len(img_name_train), len(img_name_val), len(input_tensor_train), len(input_tensor_val))

""" #cargo los embedding vectors
img_emb_train = []
print(type(img_emb_train))
for i in range(len(img_name_train)):
    #print(img_name_train[i])
    with open(img_name_train[i], 'rb') as handle:
      vec = pickle.load(handle)
      #vec = tf.reshape(pickle.load(handle)[0],[16384])
    #flat_targ = tf.reshape(vec,[vec.shape[0]*vec.shape[1]])
    #print(flat_targ)
    img_emb_train.append(vec)
    if i%100==0:
        print(vec.shape)
        #print(flat_targ.shape)
        print(i)  """

BUFFER_SIZE = 1000
BATCH_SIZE = 64
steps_per_epoch = len(img_name_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
#units = 512
output_units = 512
output_size = 16384
vocab_inp_size = top_k + 1 #ojooo
#vocab_inp_size = len(inp_lang.word_index)+1 #VER SI ESTA BIEN

# Load the numpy files
#def map_func( cap,img_name):
#  with open(img_name.decode('utf-8'), 'rb') as handle:
#    img_tensor = pickle.load(handle)
#  print("pickle shape",img_tensor.shape)
#  #img_tensor = np.load(img_name.decode('utf-8')+'.npy')
#  return  cap,img_tensor
# Load the numpy files
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap

dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
#print ("data",len(input_tensor_train),input_tensor_train[0].shape,len(img_emb_train),img_emb_train[0].shape)
#dataset = tf.data.Dataset.from_tensor_slices(( img_name_train,img_name_train))
#dataset = tf.data.Dataset.from_tensor_slices(( input_tensor_train,img_emb_train))
#print(dataset.shape)
# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

#asi lo hacia el translator
#dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
#dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

#example_input_batch, example_target_batch = next(iter(dataset))
#example_input_batch.shape, example_target_batch.shape
#agregado por Aldo

#fin agregado por Aldo

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
    output_gru, state = self.gru(x, initial_state = hidden)

    #print (output_gru)
    #print (state)
    cnn1out = self.cnn1(output_gru)
    #print (cnn1out)
    flat_vec = tf.reshape(cnn1out,[cnn1out.shape[0],cnn1out.shape[1]*cnn1out.shape[2]])
    #flat_vec = tf.keras.layers.MaxPooling1D(pool_size=8,strides=4, padding='same')(flat_vec)
    #print(flat_vec.shape)
    #print(flat_vec)
    #flat_vec = tf.reshape(flat_vec,[flat_vec.shape[0],flat_vec.shape[1]])
    #print(flat_vec.shape)
    #sys.exit()

    output = self.fc(flat_vec)
    #print(output)
    #sys.exit()
    #output = self.fc(output_gru)
    output = tf.nn.relu(output) #ver si reduce el error
    #print(output[0][0].numpy())
    return output, state
    #return flat_vec, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))


print("parametros encoder",vocab_inp_size, embedding_dim, units, output_size, BATCH_SIZE)
#sys.exit(0)
print("va con relu")
encoder = Encoder(vocab_inp_size, embedding_dim, units, output_size, BATCH_SIZE)

# sample input
#sample_hidden = encoder.initialize_hidden_state()
#sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
#print ('Encoder input shape: (batch size, sequence length) {}'.format(example_input_batch.shape))
#print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
#print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))



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
  #print(loss_)
  #sys.exit()
  #return loss_

""" checkpoint_dir = './training_emb_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(optimizer=optimizer,
#                                 encoder=encoder,
#                                 decoder=decoder) 
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder)      """
                                 
#checkpoint_path = './training_emb_checkpoints_pru'
checkpoint_path = './ckk_pru'

ckpt = tf.train.Checkpoint(encoder=encoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  # restoring the latest checkpoint in checkpoint_path
  ckpt.restore(ckpt_manager.latest_checkpoint)                           
#max_poll1D = tf.keras.layers.MaxPool1D(pool_size=2, strides=None, padding='valid')
@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0
 # targ = max_poll1D(targ)
  #print(targ.shape)
  #sys.exit()
  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    #dec_hidden = enc_hidden

    #dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    #for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      #predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

    #loss += loss_function(targ[:, t], predictions)
    #!loss += loss_function(targ[:,targ.shape[1]], enc_output)
    #print(targ)
    #print(enc_output)
    #sys.exit()
    loss += loss_function(targ,enc_output)
    #print (targ)
    #print (enc_output)
    #loss += loss_function(targ[:, t], enc_output)
    #loss = 1

      # using teacher forcing
    #  dec_input = tf.expand_dims(targ[:, t], 1)

  # batch_loss = (loss / int(targ.shape[1]))
  batch_loss = loss

  variables = encoder.trainable_variables #+ decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss,enc_output

EPOCHS = 500
print( "v41")
for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0
  print("llegue")
  #dd = dataset.take(steps_per_epoch)
  #print (type(dd))
  #print(steps_per_epoch)


  #for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):

  for (batch, (targ,cap)) in enumerate(dataset):
    #verrr
    enc_hidden = encoder.initialize_hidden_state()
    batch_loss,res = train_step(cap, targ, enc_hidden)
    print(res)
    print(res[50].numpy())
    print(targ[50].numpy())
    sys.exit()

    total_loss += batch_loss

    if batch % 100 == 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))  

  if epoch % 2 == 0:
      ckpt_manager.save()
  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))