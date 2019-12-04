#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:39:50 2019

@author: seungmin
"""

import os
import numpy as np

import requests
import json

def csv_data_loader(path):
    # 확장자 예외처리
    if os.path.splitext(path)[-1].lower() == '.csv':
        array = np.genfromtxt(path, delimiter=',')
    print("{}".format(array.shape))

def training_data_shaper(my_array):
    return my_array


def realtime_api_json(url):
    
    response = requests.get(url).text
    response_array = json.loads(response)
    
    return response_array

def realtime_data_shaper(my_array):
    assert my_array.shape != (100, 30) # 데이터 shape 예외처리
    
    # null값 처리
    
    # 이상치 처리
    
    return my_array
    