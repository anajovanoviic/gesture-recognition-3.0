# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 12:21:27 2025

@author: anadjj
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 13:40:03 2025

@author: anadjj
"""

from keras.models import Sequential
from keras.layers import Dense
from numpy import loadtxt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
#from random import shuffle
from numpy.random import shuffle
from utils import save_model

import numpy as np
from keras import models


"""Sigmoid function is used for binary classification. Softmax is used for 
multi-class scenarios.
"""

np.random.seed(7)

def split_data(data, test_data, mix, test_rate):
    
    if mix == True:
        shuffle(data)
        shuffle(test_data)
    test_num = int(len(data)*(1 - test_rate))
    
    x_train = data[:test_num, 1:43]
    y_train = data[:test_num, 0]
    x_test = test_data[:, 1:43]
    y_test = test_data[:, 0]
    
    return x_train, y_train, x_test, y_test
    

dataset = loadtxt('C:\\Users\\anadjj\\programs_ana\\master_thesis_final\\gesture-recognition\\dataset_8.csv', delimiter=',')
print(dataset)

test_dataset = loadtxt('C:\\Users\\anadjj\\programs_ana\\master_thesis_final\\gesture-recognition\\dataset_8_test_Suza.csv', delimiter=',')

x_train, y_train, x_test, y_test = split_data(dataset, test_dataset, True, 0.2)

model = models.load_model("C:\\Users\\anadjj\\programs_ana\\master_thesis_final\\gesture-recognition\\mediapipe\\eight_gestures\\model_8.h5")

#model = models.load_model('model_8.h5')

_, accuracy = model.evaluate(x_train, y_train)
print('Train accuracy: %.2f' % (accuracy*100))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nTest accuracy: {test_acc}')
