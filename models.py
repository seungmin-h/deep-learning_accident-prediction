#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:50:56 2019

@author: seungmin
"""

import tensorflow as tf
from tf.keras.models import Sequential
from tf.keras.layers import Dense, Activation
from tf.keras.optimizers import SGD


def my_classification_model(my_units, my_dim, lr):
    model = Sequential()
    
    model.add(Dense(units = int(my_units),
                    input_dim = int(my_dim)*int(my_dim),
                    activation = 'relu'))
    # ...
    
    return model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr))

def my_regression_model(my_units, my_dim, lr):
    model = Sequential()
    
    # ...
    return model.compile(loss='mean_squared_error', optimizer=SGD(lr=lr))