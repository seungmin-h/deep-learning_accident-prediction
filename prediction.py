#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:17:24 2019

@author: seungmin
"""

from models import *
from preprocessor import *

import requests
import json

import argparse
import time
import datetime

import tensorflow as tf
from tf.keras.layers import Dense
from tf.keras.models import load_model


# 커맨드 라인 입력 변수
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "교통사고 예측 모듈 v1.0.0")
    
    parser.add_argument("-w", "--weights_path", type=str, default="weights/my_model.h5",
                        help=": 학습이 끝난 가중치 파일의 경로를 입력하세요.")
    
    parser.add_argument("-api", "--api_key", type=str, default=False,
                        help=": 학습이 끝난 가중치 파일의 경로를 입력하세요.")
    
    opt = parser.parse_args()
    print(opt)
    
# 가중치 초기화
my_model = load_model(opt.weights_path)
print("가중치 파일을 업로드 하였습니다.")

# 실시간 api 예상 
rt_api = time.time()
url_01 = "https://......" + "&api_key=" + opt.api_key
rt_tensor = realtime_data_shaper(rt_tensor)

# 정규화 
norm_rt_tensor = rt_tensor / np.linalg.norm(rt_tensor) 

# 추론
inf_tensor = my_model.predict(norm_rt_tensor)
rt_inf = datetime.datetime.now().strftime("%H:%M:%S")
print("{0}--{1}".format(realtime, inf_tensor.shape))

# 반정규화
