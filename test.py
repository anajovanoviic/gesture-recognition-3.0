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

#np.random.seed(7)
#np.random.seed(42)
np.random.seed(0) #najbolje za sada
#np.random.seed(123)
#np.random.seed(99)

def split_data(data, mix, test_rate):
    
    if mix == True:
        shuffle(data)
    test_num = int(len(data)*(1 - test_rate))
    
    x_train = data[:test_num, 1:43]
    y_train = data[:test_num, 0]
    x_test = data[test_num:, 1:43]
    y_test = data[test_num:, 0]
    
    return x_train, y_train, x_test, y_test
    

dataset = loadtxt('C:\\Users\\anadjj\\programs_ana\\master_thesis_final\\gesture-recognition\\dataset_8.csv', delimiter=',')
#dataset = loadtxt('C:\\Users\\anadjj\\programs_ana\\master_thesis_final\\gesture-recognition\\dataset_8_both_hands.csv', delimiter=',')
#print(dataset)

x_train, y_train, x_test, y_test = split_data(dataset, True, 0.2)

model = models.load_model("C:\\Users\\anadjj\\programs_ana\\master_thesis_final\\gesture-recognition\\mediapipe\\eight_gestures\\model_8.h5")
#model = models.load_model("C:\\Users\\anadjj\\programs_ana\\master_thesis_final\\gesture-recognition\\refactored\\model_8_both_hands.h5")
#model = models.load_model("C:\\Users\\anadjj\\programs_ana\\master-thesis-code-github-1-11-2025\\gesture-recognition\\model_8_both_hands_17_1_22h.h5")

#model = models.load_model('model_8.h5')

_, accuracy = model.evaluate(x_train, y_train)
print('Train accuracy: %.2f' % (accuracy*100))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nTest accuracy: {test_acc}')