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

""" print(all_captions[55117])
print(all_captions[55112])
print(all_captions[55431])
print(all_captions[55588])
print(all_captions[55493])
 """


""" <start> a motorcycle that is parked next to a street <end>
<start> a motorcycle parked on a brick street with a sky background <end>
<start> A close of of the font end of a motorcycle. <end>
<start> A white, double seated motorcycle is Parked by the curb <end>
<start> A large motorcycle sits outside of a bar. <end>
<start> A motorcycle parked on the side of a street. <end> """

#fileTXT = './img_embeddings/encodedText_000000201278_000000128780.emdt'
#fileTXT = './img_embeddings/encodedText_000000538931_00000046682.emdt'
#fileTXT ='./img_embeddings/encodedText_000000538931_000000219917.emdt'
fileTXT = './img_embeddings/encodedText_000000001232_000000043564.emdt'
with open(fileTXT, 'rb') as handle:
    txt_tensor = pickle.load(handle)
#error = loss_object(txt_tensor,img_tensor)

loss_object = tf.keras.losses.MeanSquaredError()
#files=glob.glob('./img_embeddings/encodedText_000000493717_*.emdt')
files=glob.glob('./img_embeddings/COCO_train2014_*.npy')
min_error_img_tensor = 1000000
min_error_img_tensor_file = "No encontre"

thebest = []

for i in range(len(files)):
    if (i %1000 == 0) or (i %500 == 0 and i<5000) or (i%100 == 0 and i<1000):
        print ("NUEVO")
        for s in range(len(thebest)):
            print (i,s,thebest[s])
    #./img_embeddings/encodedText_000000493717_000000055117.emdt
    img_tensor = np.load(files[i])

    error = loss_object(txt_tensor,img_tensor)
    if len(thebest) < 10:
        thebest.append([min_error_img_tensor,min_error_img_tensor_file])
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


