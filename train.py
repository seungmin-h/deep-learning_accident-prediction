#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:24:46 2019

@author: seungmin
"""


import argparse

from tensorflow import keras
from keras.models import load_model
from sklearn.model_selection import train_test_split

from models import *
from preprocessor import *

# 커맨드 라인 입력 변수
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "교통사고 학습 모듈 v1.0.0")
    
    parser.add_argument("-d", "--data_path", type=str, default="data/my_data.csv",
                        help=": 과거 학습데이터 파일의 경로를 입력하세요.")
    
    parser.add_argument("-s", "--save_weights", type=str, default="weights/my_model.h5",
                        help=": 학습이 끝난 가중치가 저장될 경로를 입력하세요.")
    
    parser.add_argument("-n", "--units", type=int, default=32,
                        help=": my unit")
    
    opt = parser.parse_args()
    print(opt)
    
# 과거 데이터 전처리
input_array = csv_data_loader(opt.data_path)
input_array = training_data_shaper(train_input)

# normalization + 테스트 트레인 스플릿 
norm_input_array = input_array / np.linalg.norm(input_array)
x_train, x_valid, y_train, y_valid = train_test_split() ### 


# 파라미터, 최적화 함수 설정
my_dim = 28
lr = 0.001
batch_size = 64

# 모델 초기화
my_model = my_classification_model(opt.units, my_dim, lr)

# 학습 시작
my_model.fit(x_train, y_train, batch_size=) ###
print("\nAccuracy: {:.3f}".format(model.evaluate()[1])) ## 텐서플로우 로거 대체

# 체크포인트 등 상세 디자인 생략

my_model.save(opt.save_weights)
print("학습이 끝났습니다.")