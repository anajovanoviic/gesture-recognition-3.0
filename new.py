# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 11:49:24 2024

@author: anadjj
"""

# next three lines are with copilot
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from numpy import loadtxt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
#from random import shuffle
from numpy.random import shuffle
from utils import save_model

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random

"""Sigmoid function is used for binary classification. Softmax is used for 
multi-class scenarios.
"""

SEED = 7

def split_data(data, mix, test_rate):
    
    if mix == True:
        shuffle(data)
    test_num = int(len(data)*(1 - test_rate))
    
    x_train = data[:test_num, 1:43]
    y_train = data[:test_num, 0]
    x_test = data[test_num:, 1:43]
    y_test = data[test_num:, 0]
    
    return x_train, y_train, x_test, y_test

def plot_history(history):
    """
    Plot training and validation accuracy and loss
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['sparse_categorical_accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_sparse_categorical_accuracy'], label='Test Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Test Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
def neural_network(x_train, y_train, x_test, y_test):
    
    model = Sequential()
    model.add(Dense(20, input_dim=42, activation='relu'))
    model.add(Dense(20, activation='relu')) # when I removed this layer I got test accuracy less then 1
    model.add(Dense(10, activation='relu'))
    model.add(Dense(8, activation='softmax')) 
    
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(optimizer=Adam(),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=[SparseCategoricalAccuracy()])
    
    model.summary()
    
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
    
    return model, history

def main():
    
    # Set seeds at the beginning of main to ensure reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    dataset = loadtxt('C:\\Users\\anadjj\\programs_ana\\master_thesis_final\\gesture-recognition\\dataset_8_both_hands.csv', delimiter=',')
    print(dataset)
    
    x_train, y_train, x_test, y_test = split_data(dataset, True, 0.2)
    
    model, history = neural_network(x_train, y_train, x_test, y_test)
    
    # Plot training history
    plot_history(history)
    
    _, accuracy = model.evaluate(x_train, y_train)
    print('Train accuracy: %.2f' % (accuracy*100))
    
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'\nTest accuracy: {test_acc}')
    
    filename="model_8_both_hands_19_1_v3.h5"
    save_model(model, filename)
    
    y_pred = model.predict(x_test)
    #for i in range(32):
    #    print("X=%s, Predicted=%s" % (x_test[i], y_pred[i]))
        

if __name__ == "__main__":
    main()