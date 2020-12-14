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
from glob import glob
from PIL import Image
import pickle
import Dataset as ds

# Path para cargar el checkpoint del encoder y evaluar                         
checkpoint_path = '/workspace/checkpoints/encoder_text/'

# Path para guardar la salida del codificador
encoded_captions_path = '/workspace/pickle_saves/encoded_eval_captions/'
#encoded_captions_path = '/workspace/pickle_saves/encoded_train_captions/'

LOG_PATH = './Evaluations/logs/'

# Parametros de entrenamiento
EPOCHS = 100
MAX_CKPT = 10    #max checkpoints before overwrite
CHPK_SAVE = 10   #save between epochs
EPOCHS = 100     #training epochs

# Parametros del modelo
BATCH_SIZE = 64
embedding_dim = 256
units = 1024
vocab_inp_size = 5001
output_units = 512
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
            self.gru = tf.keras.layers.GRU(self.enc_units,
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform')  # capa GRU
            self.cnn1 = tf.keras.layers.Conv1D(256, 4, activation='relu',input_shape = [49, 1024])  # capa Convolucional
            self.fc = tf.keras.layers.Dense(enc_output_units)   # FC1 - Dense ------(if no activation is specified it uses linear)
  

    def call(self, x, hidden):
        x = self.embedding(x)   # capa Embedding
        output_gru, state = self.gru(x, initial_state = hidden) # capa GRU
        cnn1out = self.cnn1(output_gru) # capa CNN
        flat_vec = tf.reshape(cnn1out,[cnn1out.shape[0],cnn1out.shape[1]*cnn1out.shape[2]]) # Flattened
        output = self.fc(flat_vec)   # capa densa de salida - FC
        output = tf.nn.relu(output) #ver si reduce el error
        return output, state # output shape (16384,)

    # Inicio hidden state todo en cero
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units)) 

    # Obtengo ENCODER
    encoder = Encoder(vocab_inp_size, embedding_dim, units, output_size, BATCH_SIZE)

    return encoder

#Carga el ultimo checkpoint
def load_checkpoint(encoder):

    optimizer = tf.keras.optimizers.Adam()

    # Creamos objeto checkpoint para el encoder y decoder ya entrenado previamente
    ckpt = tf.train.Checkpoint(encoder=encoder,optimizer = optimizer)

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

def evaluate(encoder,caption):
  enc_hidden = encoder.initialize_hidden_state()
  enc_output, enc_hidden = encoder(caption, enc_hidden) 
  return enc_output

def generate_embeddings(encoder,dataset):
    #Codifica las captions de captions_to_encode y los guarda en encoded_captions_path
    text_id = 0
    print("------------------ Generating encoded captions and saving them in %s ----------------\n" % (encoded_captions_path) )
    for (batch,(caption,img_name)) in enumerate(dataset):
        encoded_captions_batch = evaluate(encoder,caption) # salida del encoder
        #print("caption : %s \n encoded_caption shape : %s \n" % (caption[0],text_encoded_vec[0]))
        for i in range(64): 
            text_id += 1
            #print(("i = %d , img_name : %s \n")%(i,img_name[i]))
            image_name = str(img_name[i])
            img_id = int(image_name.rsplit('/',1)[1].split('_')[2][:12])
            full_encoded_captions_path = encoded_captions_path + 'encodedText_%012d_%012d.emdt' % (img_id,text_id)
            with open(full_encoded_captions_path, 'wb') as handle:
                pickle.dump(encoded_captions_batch[i].numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL) 
        if batch % 10==0:
            print("batch %d \n",batch)

## Funcion de calculo de la perdida segun tensor real y predecido
def loss_function(loss_object,real, pred):
  loss_ = loss_object(real, pred)
  return tf.reduce_mean(loss_)

def read_train_log(train_log_path):
    f = open(train_log_path, "r")
    linea = f.readlines()
    train_losses = []
    indice = 0
    for renglon in linea:
        Train_line = re.search(r"Epoch\s(\d+)\sLoss\s(\d+.\d+)", renglon)
        if Train_line is not None:
            train_losses.append( [float(Train_line.group(2)),int(Train_line.group(1))] )
    return train_losses

# Plot training losses from log and the evalutaion losses of encoder text checkpoints
def plot_eval_losses(encoder,dataset_eval,train_losses):
    eval_losses = get_eval_losses(encoder,dataset_eval)
    #get min train loss
    min_train_loss = 100
    for i,loss,epoch in enumerate(train_losses):
        if(loss < min_train_loss):
            min_train_loss = loss
            min_train_epoch = epoch
        print("Epoch: %d -- Train Loss: %f Evalutaion Loss : %f \n" % (epoch,loss,eval_losses[i][1]))
    print("Minimum training loss at epoch %d : %f\n" % (min_train_epoch,min_train_loss) )

    ## Generar grafico y mostrar
    plot.plot(train_losses[][0],train_Loss[][1],'go--',label='Training Loss')
    plot.plot(eval_losses[][0],eval_losses[][1],'ro--',label='Evalutaion Loss')
    plot.legend(loc='lower left')
    plot.xlabel("Epoch")
    plot.ylabel("Losses")
    plot.title("Losses in encoder text training")
    plot.rcParams["figure.figsize"] = (15,10)
    plot.savefig(LOG_PATH + 'encoder_text_loss_plot.png')
    print("Plot saved succesfully in %s \n" % (LOG_PATH) )

# Load all checkpoints and calculate eval loss for each one
def get_eval_losses(encoder,dataset_eval):
  eval_loss = []
  eval_loss_epochs = []
  print("------- Starting evaluation of encoder text checkpoints -------\n")
  eval_loss.append([float(get_avg_loss(encoder,dataset_eval)),0])
  print("Evaluation loss epoch (%d) : %s \n" % (eval_loss_epochs[0],eval_loss[0]) )
  for i in range(10):
    checkpoint_name = checkpoint_path + 'ckpt-' + str(10*(i+1))
    tf.print("----------- Restoring from {} -----------".format(checkpoint_name))
    ckpt.restore(checkpoint_name)
    eval_loss.append([float(get_avg_loss()),int(10*(i+1))])
    eval_loss_epochs.append(int(10*(i+1)))
    print("Evaluation loss epoch (%d) : %s \n" % (eval_loss_epochs[i+1],eval_loss[i+1]) )
  return eval_losses

def get_batch_loss(encoder,captions_batch,enc_images_batch):
  enc_hidden = encoder.initialize_hidden_state()
  enc_output, enc_hidden = encoder(captions_batch, enc_hidden)
  batch_loss = 0
  for i in range (BATCH_SIZE):
    batch_loss += loss_function(enc_output[i],enc_images_batch[i])
  return (batch_loss/BATCH_SIZE)

def get_avg_loss(encoder,dataset_eval):
  total_loss = 0
  for (batch,(enc_images_batch,captions_batch) ) in enumerate(dataset_eval):
    total_loss+=get_batch_loss(encoder,captions_batch,enc_images_batch)
  return (total_loss/batch)

# Encoder text training
def train(start_epoch,encoder,dataset_train,ckpt_manager,optimizer):
    
    print("\n-------------  Starting %d epoch's training for text encoder model  ------------\n"% (EPOCHS) )
    print("Saving model every %d epochs at %s\n"%(CHPK_SAVE,checkpoint_path))
    train_losses = []
    #print('Evaluation loss : %f \n' %(get_avg_loss()))

    for epoch in range(start_epoch , EPOCHS):
        start = time.time() # inicio cuenta de tiempo
        enc_hidden = encoder.initialize_hidden_state() # inicio hidden state en cero
        total_loss = 0 # reinicio cuenta loss

        for (batch, (targ,cap)) in enumerate(dataset_train): #iterate over dataset (encoded image(.emb) (target), caption (input))
            #verrr
            enc_hidden = encoder.initialize_hidden_state()
            batch_loss,res = train_step(cap, targ, enc_hidden,optimizer)
            #print(res[50].numpy())  #imprime un elemento cualquiera del tensor del caption con el valor real del target para comparar
            #print(targ[50].numpy())
            #sys.exit()

            total_loss += batch_loss 

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,batch,batch_loss.numpy()))  

        # computo loss de cada epoch
        train_losses.append([total_loss / BATCH_SIZE ,epoch+1])

        if (epoch!=0) and (epoch % CKPT_EPOCH_SAVE ) == 0:
            ckpt_manager.save(checkpoint_number=epoch)   ## almaceno checkpoint cada 2 epoch's
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    return train_losses

@tf.function
def train_step(inp, targ, enc_hidden,optimizer):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        loss += loss_function(targ,enc_output)
    batch_loss = loss 
    variables = encoder.trainable_variables 
    gradients = tape.gradient(loss, variables) 
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss,enc_output  

encoder=get_model()
dataset_train,dataset_eval = ds.get_enc_image_caption_datasets()
ckpt_manager,optimizer,start_epoch = load_checkpoint(encoder)
train_losses = train(start_epoch,encoder,dataset_train,ckpt_manager,optimizer)

