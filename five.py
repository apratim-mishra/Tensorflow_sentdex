import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

import pickle

import time

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

X = X/255.0

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
y = np.array(y)

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='C:/Users/Imp/Microsoft/Sentdex/new_ML/logs/{}'.format(NAME))
            print(NAME)
            
            model1 = Sequential()
            model1.add(Conv2D(layer_size, kernel_size=(3,3), activation='relu', input_shape = X.shape[1:]))
            model1.add(MaxPooling2D(pool_size = (2,2)))

            for l in range(conv_layer-1):

                model1.add(Conv2D(layer_size), kernel_size=(3,3), activation='relu'))
                model1.add(MaxPooling2D(pool_size = (2,2)))
            
            model1.add(Flatten())
            for l in range(dense_layer):
                               
                model1.add(Dense(layer_size))
                model1.add(Activation('relu'))

            model1.add(Dense(1))
            model1.add(Activation('sigmoid'))

            model1.compile(loss='binary_crossentropy', 
                        optimizer='adam', 
                        metrics=['accuracy'])

            model1.fit(X, y, batch_size = 32, validation_split = 0.1, epochs = 3, callbacks = [tensorboard])








