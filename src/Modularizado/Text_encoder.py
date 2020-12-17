## -- ENCODER TEXT
import tensorflow as tf
import re
import matplotlib.pyplot as plot
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
import glob
from PIL import Image
import pickle
import Dataset as ds

# Path para guardar/cargar el checkpoint del encoder                        
#checkpoint_path = '/workspace/checkpoints/checkpoints_80%COCO/encoder_text_100epochs_2.5loss'
checkpoint_path = '/workspace/checkpoints/encoder_text_noConv/'

# Log de lectura para recuperar train losses
TRAIN_LOSSES_LOG = './Evaluations/logs/encoder_text_noConvol.txt'
# Path para guardar plot de losses
LOSSES_PLOT = './Evaluations/'
#TRAIN_LOSSES_LOG = './Evaluations/logs/encoder_text_100epoch.txt'

# Path para guardar la salida del codificador
encoded_captions_path = '/workspace/pickle_saves/encoded_eval_captions_noconvol/'
#encoded_captions_path = '/workspace/pickle_saves/encoded_train_captions/'

# Parametros de entrenamiento
EPOCHS = 20
MAX_CKPT = 10    #max checkpoints before overwrite
CHPK_SAVE = 2   #save between epochs

# Parametros del modelo
BATCH_SIZE = 64
embedding_dim = 256
units = 256 # 1024
vocab_inp_size = 5001
output_size = 16384 

# Instanciacion del modelo encoder text
def get_model():

    class Encoder(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, enc_units, enc_output_units,batch_sz):
            super(Encoder, self).__init__()
            self.batch_sz = batch_sz
            self.enc_units = enc_units
            self.enc_output_units = enc_output_units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) # capa Embedding
            self.gru = tf.keras.layers.GRU(self.enc_units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')  # capa GRU
            self.cnn1 = tf.keras.layers.Conv1D(256, 4, activation='relu',input_shape = [52,256])  # capa Convolucional
            self.fc = tf.keras.layers.Dense(enc_output_units)   # FC1 - Dense ------(if no activation is specified it uses linear)

        def call(self, x, hidden):
            x = self.embedding(x)   # capa Embedding
            # (64, 52, 256) (batch_size,max_caption_len,emb_dimension)
            output_gru, state = self.gru(x, initial_state = hidden) # capa GRU
            # (64,52,256)
            #cnn1out = self.cnn1(output_gru) # capa CNN
            # (64,49,256)
            flat_vec = tf.reshape(output_gru,[output_gru.shape[0],-1])
            #flat_vec = tf.reshape(cnn1out,[cnn1out.shape[0],cnn1out.shape[1]*cnn1out.shape[2]]) # Flattened
            #(64, 12544)
            output = self.fc(flat_vec)   # capa densa de salida - FC
            output = tf.nn.relu(output) #ver si reduce el error
            #(64, 16384)
            return output, state

        # Inicio hidden state todo en cero
        def initialize_hidden_state(self):
            return tf.zeros((self.batch_sz, self.enc_units)) 

    # Obtengo ENCODER
    encoder = Encoder(vocab_inp_size, embedding_dim, units, output_size, BATCH_SIZE)

    return encoder

#Carga el ultimo checkpoint
def load_checkpoint(encoder,ckpt_name):

    optimizer = tf.keras.optimizers.Adam()

    # Creamos objeto checkpoint para el encoder y decoder ya entrenado previamente
    ckpt = tf.train.Checkpoint(encoder=encoder,optimizer = optimizer)

    # Establezco el checkpoint manager a usar (limite para 5 ultimos checkpoints)
    ckpt_manager = tf.train.CheckpointManager(ckpt,checkpoint_path, max_to_keep=MAX_CKPT)

    start_epoch = 0
    if (ckpt_name is None):
        if ckpt_manager.latest_checkpoint: # si existen checkpoints
            start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
            # restoring the latest checkpoint in checkpoint_path
            ckpt.restore(ckpt_manager.latest_checkpoint)  # cargo el ultimo checkpoint disponible  
            print("Restored from {}".format(ckpt_manager.latest_checkpoint))
        else:
            print("----- No checkpoints found in %s -----\n----- initializing training from scratch ----- \n"% (checkpoint_path))
    else:
        ckpt.restore(checkpoint_path + ckpt_name).expect_partial()
        start_epoch = int(ckpt_name.split('-')[-1])
        print("Restored from %s \n" % (checkpoint_path+ckpt_name) )

    return encoder,ckpt_manager,optimizer,start_epoch

def evaluate(encoder,caption):
  enc_hidden = encoder.initialize_hidden_state()
  enc_output, enc_hidden = encoder(caption, enc_hidden) 
  return enc_output

def generate_embeddings(encoder,captions_list,img_names_list):
    #Codifica las captions de captions_to_encode y los guarda en encoded_captions_path
    text_id = 0
    dataset = tf.data.Dataset.from_tensor_slices((captions_list,img_names_list)).batch(BATCH_SIZE,drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    print("------------------ Generating encoded captions and saving them in %s ----------------\n" % (encoded_captions_path) )
    for (batch,(caption,img_name)) in enumerate(dataset):
        encoded_captions_batch = evaluate(encoder,caption) # salida del encoder
        #print("caption : %s \n encoded_caption shape : %s \n" % (caption[0],text_encoded_vec[0]))
        for i in range(64):
            text_id += 1
            image_name = str(img_name[i])
            img_id = int(image_name.split('_')[2][:12])
            full_encoded_captions_path = encoded_captions_path + 'encodedText_%012d_%012d.emdt' % (img_id,text_id)
            with open(full_encoded_captions_path, 'wb') as handle:
                pickle.dump(encoded_captions_batch[i].numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL) 
        if batch % 10==0:
            print("Succesuflly encoded batch %d \n" % batch)

## Funcion de calculo de la perdida segun tensor real y predecido
def loss_function(loss_object,real, pred):
  loss_ = loss_object(real, pred)
  return tf.reduce_mean(loss_)

def read_train_log():
    f = open(TRAIN_LOSSES_LOG, "r")
    linea = f.readlines()
    train_losses = []
    indice = 0
    for renglon in linea:
        Train_line = re.search(r"Epoch\s(\d+)\sLoss\s(\d+.\d+)", renglon)
        if Train_line is not None:
            train_losses.append([float(Train_line.group(2)),int(Train_line.group(1))])
    return train_losses

# Plot training losses from log and the evalutaion losses of encoder text checkpoints
def plot_losses(encoder,dataset_eval,plot_name):
    train_losses = read_train_log()
    eval_losses = get_eval_losses(encoder,dataset_eval)
    #get min train loss
    min_train_loss = 100
    i = 0
    for loss,epoch in train_losses:
        if(loss < min_train_loss):
            min_train_loss = loss
            min_train_epoch = epoch
        print("Epoch: %d -- Train Loss: %f \n" % (epoch,loss))
        i+=1
    print("Minimum training loss at epoch %d : %f\n" % (min_train_epoch,min_train_loss) )

    def get_elements(lst,index):
        return [item[index] for item in lst]

    ## Generar grafico y mostrar
    plot.plot(get_elements(train_losses,1),get_elements(train_losses,0),'go--',label='Training Loss')
    plot.plot(get_elements(eval_losses,1),get_elements(eval_losses,0),'ro--',label='Evalutaion Loss')
    plot.legend(loc='lower left')
    plot.xlabel("Epoch")
    plot.ylabel("Losses")
    plot.title("Losses in encoder text training")
    plot.rcParams["figure.figsize"] = (15,10)
    plot.savefig(LOSSES_PLOT + plot_name)
    print("Plot saved succesfully in %s \n" % (LOSSES_PLOT) )

# Load all checkpoints and calculate eval loss for each one
def get_eval_losses(encoder,dataset_eval):

  eval_loss = []
  loss_object = tf.keras.losses.MeanSquaredError()
  checkpoint_list =glob.glob(checkpoint_path + '*.index') 
  checkpoint_list = sorted(checkpoint_list,key= lambda i:int(i.split('/')[-1].split('.')[0].split('-')[1]))

  print("------- Starting loss evaluation of encoder text checkpoints -------\nFounded %d checkpoints\n"% (len(checkpoint_list)))
  eval_loss.append([float(get_avg_loss(encoder,dataset_eval,loss_object)),0])
  print("Evaluation loss at epoch (%d) : %f \n" % (0,eval_loss[0][0]) )

  for ckpt in checkpoint_list:
    encoder,ckpt_manager,optimizer,start_epoch = load_checkpoint(encoder,ckpt.split('/')[-1].split('.')[0])
    eval_loss.append([float(get_avg_loss(encoder,dataset_eval,loss_object)),start_epoch])
    print("Evaluation loss at epoch %d : %f \n" % (start_epoch,eval_loss[-1][0]))

  return eval_loss

def get_batch_loss(encoder,captions_batch,enc_images_batch,loss_object):
  enc_hidden = encoder.initialize_hidden_state()
  enc_output, enc_hidden = encoder(captions_batch,enc_hidden)
  batch_loss = 0
  for i in range (BATCH_SIZE):
    batch_loss += loss_function(loss_object,enc_output[i],enc_images_batch[i])
  return (batch_loss/BATCH_SIZE)

def get_avg_loss(encoder,dataset_eval,loss_object):
  total_loss = 0
  for (batch,(captions_batch,enc_images_batch)) in enumerate(dataset_eval):
    total_loss+=get_batch_loss(encoder,captions_batch,enc_images_batch,loss_object)
    if batch!=0 and batch%1200==0:
        break
  return (total_loss/batch)

# Encoder text training
def train(start_epoch,encoder,dataset_train,ckpt_manager,optimizer):
    
    print("\n-------------  Starting %d epoch's training for text encoder model  ------------\n"% (EPOCHS) )
    print("Saving model every %d epochs at %s\n"%(CHPK_SAVE,checkpoint_path))
    train_losses = []
    loss_object = tf.keras.losses.MeanSquaredError()
    #print('Evaluation loss : %f \n' %(get_avg_loss()))

    for epoch in range(start_epoch , EPOCHS):
        start = time.time() # inicio cuenta de tiempo
        enc_hidden = encoder.initialize_hidden_state() # inicio hidden state en cero
        total_loss = 0 # reinicio cuenta loss

        for (batch, (cap,targ)) in enumerate(dataset_train): #iterate over dataset (encoded image(.emb) (target), caption (input))
            enc_hidden = encoder.initialize_hidden_state()
            batch_loss,res = train_step(cap, targ, enc_hidden,loss_object,optimizer)
            total_loss += batch_loss 

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,batch,batch_loss.numpy())) 

            if (batch!=0) and (batch%5100==0):
                break 

        # computo loss de cada epoch
        train_losses.append([total_loss / BATCH_SIZE ,epoch+1])

        if  ( (epoch+1) % CHPK_SAVE ==0) :
            ckpt_manager.save(checkpoint_number=epoch+1)  
            print("Checkpoint %d saved at %s\n" % (epoch+1,checkpoint_path) )
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / batch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    return train_losses

@tf.function
def train_step(inp, targ, enc_hidden,loss_object,optimizer):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        loss += loss_function(loss_object,targ,enc_output)
    batch_loss = loss 
    variables = encoder.trainable_variables 
    gradients = tape.gradient(loss, variables) 
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss,enc_output  

#dataset_train,dataset_eval = ds.get_enc_image_caption_datasets()
#images_names_train, images_names_eval , captions_train, captions_eval , tokenizer  = ds.get_image_names_captions_lists()
#encoder=get_model()
#encoder,ckpt_manager,optimizer,start_epoch = load_checkpoint(encoder,'ckpt-2')
#generate_embeddings(encoder,captions_eval,images_names_eval)
#plot_losses(encoder,dataset_eval,'encoder_text_noConvol')
#train_losses = train(start_epoch,encoder,dataset_train,ckpt_manager,optimizer)

