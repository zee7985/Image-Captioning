import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import pickle
import json
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add

model=load_model(r"C:\Users\zeesh\Machine Learning\Image Captioning\InceptionV3\model Weights\model_38.h5")
model._make_predict_function()

model_temp = InceptionV3(weights="imagenet",input_shape=(299,299,3))

model_new = Model(model_temp.input,model_temp.layers[-2].output)
model_new._make_predict_function()

def preprocess_img(img):
    img = image.load_img(img,target_size=(299,299))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    # Normalisation
    img = preprocess_input(img)
    return img

def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_new.predict(img)
    
    feature_vector = feature_vector.reshape((1,-1))
    #print(feature_vector.shape)
    return feature_vector

with open(r"C:\Users\zeesh\Machine Learning\Image Captioning\InceptionV3\saved\word_to_idx.pkl", 'rb') as f:
    word_to_idx = pickle.load(f)

with open(r"C:\Users\zeesh\Machine Learning\Image Captioning\InceptionV3\saved\idx_to_word.pkl", 'rb') as f:    
   idx_to_word = pickle.load(f)

from math import log

def predict(image, beam_width = 3, alpha = 0.7):
  l = [('startseq', 1.0)]
  max_len=35
  for i in range(max_len):
    temp = []
    for j in range(len(l)):
      sequence = l[j][0]
      prob = l[j][1]
      if sequence.split()[-1] == 'endseq':
        t = (sequence, prob)
        temp.append(t)
        continue
      encoding = [word_to_idx[word] for word in sequence.split() if word in word_to_idx]
      encoding = pad_sequences([encoding], maxlen = max_len, padding = 'post')
      pred = model.predict([image, encoding])[0]
      pred = list(enumerate(pred))
      pred = sorted(pred, key = lambda x: x[1], reverse = True)
      pred = pred[:beam_width]
      for p in pred:
        t = (sequence + ' ' + idx_to_word[p[0]], (prob + log(p[1])) / ((i + 1)**alpha))
        temp.append(t)
    temp = sorted(temp, key = lambda x: x[1], reverse = True)
    l = temp[:beam_width]
  caption = l[0][0]
  caption = caption.split()[1:-1]
  caption = ' '.join(caption)
  return caption

def captionIt(img):
    enc=encode_image(img)
    caption =predict(enc)
    
    return caption

print(captionIt(r"C:\Users\zeesh\Machine Learning\Image Captioning\InceptionV3\images2.jfif"))


