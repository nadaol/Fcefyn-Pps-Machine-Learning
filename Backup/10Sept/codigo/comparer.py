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
import glob
max_length_set = 49
annotation_folder = '/annotations/'
image_folder = '/img_embeddings/'
max_length_set = 49
annotation_file = os.path.abspath('.') +'/annotations/captions_train2014.json'
all_all_captions = []
with open(annotation_file, 'r') as f:
    annotations = json.load(f)
#for i in annotations['annotations']:
#    if  == 55117 
#        print(annot)
with open('./tokenizer_new/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
u = 0
for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
#    print (u,caption)
    all_all_captions.append(caption)
    u+=1

all_all_captions_tok = tokenizer.texts_to_sequences(all_all_captions)
all_captions = []
all_img_name_vector  = []
for i in range(len(all_all_captions)):
    if len(all_all_captions_tok[i]) <= max_length_set:
        all_captions.append(all_all_captions[i])
#       print (len(all_captions)-1,all_captions[len(all_captions)-1])
#system.exit()
""" print(all_captions[55117])
print(all_captions[55112])
print(all_captions[55431])
print(all_captions[55588])
print(all_captions[55493])
 """

print(all_captions[43563]) #a man hanging...
sys.exit()
#print(all_captions[55116])
#print(all_captions[55111])
#print("este !!! ",all_captions[55430]) #55431
#print(all_captions[55587])
#print(all_captions[55492])
#print("ggggg ")
#print(all_captions[219916])
#print(all_captions[128779])
#print(all_captions[381238])
#print(all_captions[326283])
#print(all_captions[325959])
#print(all_captions[369572])
#print(all_captions[305931])
#print(all_captions[298313])
#print(all_captions[299540])
#print(all_captions[385775])
#print(all_captions[365417])
#sys.exit()
""" <start> a motorcycle that is parked next to a street <end>
<start> a motorcycle parked on a brick street with a sky background <end>
<start> A close of of the font end of a motorcycle. <end>
<start> A white, double seated motorcycle is Parked by the curb <end>
<start> A large motorcycle sits outside of a bar. <end>
<start> A motorcycle parked on the side of a street. <end> """

#fileTXT = './img_embeddings/encodedText_000000420051_*.emdt'
#with open(file, 'rb') as handle:
#    txt_tensor = pickle.load(handle)
#error = loss_object(txt_tensor,img_tensor)

loss_object = tf.keras.losses.MeanSquaredError()
#files=glob.glob('./img_embeddings/encodedText_000000493717_*.emdt')
files=glob.glob('./img_embeddings/*.emdt')
img_name = "./img_embeddings/COCO_train2014_000000493717"
img_name = img_name+'.npy'
img_tensor = np.load(img_name)

min_error_txt_tensor = 1000000
min_error_txt_tensor_file = "No encontre"

thebest = []

for i in range(len(files)):
    if (i %1000 == 0) or (i %500 == 0 and i<5000) or (i%100 == 0 and i<1000):
        print ("NUEVO")
        for s in range(len(thebest)):
            print (i,s,thebest[s])
    #./img_embeddings/encodedText_000000493717_000000055117.emdt
    with open(files[i], 'rb') as handle:
       txt_tensor = pickle.load(handle)

    error = loss_object(txt_tensor,img_tensor)
    if len(thebest) < 10:
        thebest.append([min_error_txt_tensor,min_error_txt_tensor_file])
    else:
        for k in range(len(thebest)):
            if error < thebest[k][0]:
                 thebest.insert(k, [error,files[i]])
                 del thebest[10]
                 break
"""     if error < min_error_txt_tensor :
        min_error_txt_tensor = error
        min_error_txt_tensor_file = files[i] """
print(thebest)

    
#print(files[i])


