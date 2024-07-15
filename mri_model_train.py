import warnings
warnings.filterwarnings('ignore')
# 경고 메시지 무시

import os
from os import listdir
import pathlib
from random import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
# 필요한 라이브러리 임포트

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import tensorflow_addons as tfa
from keras.utils import load_img, img_to_array
from keras.models import Sequential
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.layers import MaxPooling2D, Dropout, Dense, Input, Conv2D, Flatten, BatchNormalization
from keras.layers import GlobalAveragePooling2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
# 모델 학습 및 데이터 전처리를 위한 라이브러리 임포트

# 데이터 경로 설정
train_folder = 'C:/Users/RXO/Desktop/daeseong_file/project_code/Dementia/data/Alzheimer_s Dataset/train'
test_folder = 'C:/Users/RXO/Desktop/daeseong_file/project_code/Dementia/data/Alzheimer_s Dataset/test'

# 데이터 로드 예시 (디렉토리 내의 첫 번째 이미지를 로드)
def load_first_image(folder, class_name):
    class_folder = os.path.join(folder, class_name)
    image_files = os.listdir(class_folder)
    if image_files:
        first_image_path = os.path.join(class_folder, image_files[0])
        return load_img(first_image_path)
    else:
        raise FileNotFoundError(f"No images found in {class_folder}")
# 첫 번째 이미지를 로드하는 함수 정의

photo = load_first_image(train_folder, 'MildDemented')
print(photo)
photo
# MildDemented 클래스에서 첫 번째 이미지를 로드하여 출력

# 이미지 확인
plt.figure(figsize=(7,7))
j = 0
for file in os.listdir(train_folder):
    i = 0
    for image in os.listdir(train_folder + '/' + file):
        if i == 1:
            break
        img = imread(train_folder + '/' + file + '/' + image)
        ax = plt.subplot(2, 2, j+1)
        plt.imshow(img)
        plt.title(image)
        plt.axis('off')
        j = j + 1
        i = i + 1
plt.show()
# 각 클래스의 첫 번째 이미지를 시각화하여 확인

# 각 클래스별 이미지 수 세기
for file in os.listdir(train_folder):
    i = 0
    for image in os.listdir(train_folder + '/' + file):
        i = i + 1
    print(file, i)
# 각 클래스별 이미지 개수를 출력

# 이미지 데이터 생성기 설정
IMG_SIZE = 128
DIM = (IMG_SIZE, IMG_SIZE)

ZOOM = [.99, 1.01]
BRIGHT_RANGE = [0.8, 1.2]
HORZ_FLIP = True
FILL_MODE = "constant"
DATA_FORMAT = "channels_last"

train_generator = ImageDataGenerator(rescale=1./255, brightness_range=BRIGHT_RANGE, zoom_range=ZOOM, 
                                     data_format=DATA_FORMAT, fill_mode=FILL_MODE, horizontal_flip=HORZ_FLIP)
train_data_gen = train_generator.flow_from_directory(directory=train_folder, target_size=DIM, batch_size=6500, shuffle=False)
# 데이터 증강을 위한 ImageDataGenerator 설정 및 생성

CLASSES = list(train_data_gen.class_indices.keys())
# 클래스 이름 저장

def show_images(generator, y_pred=None):
    # 이미지 라벨 가져오기
    labels = dict(zip([0, 1, 2, 3], CLASSES))
    
    # 이미지 배치 가져오기
    x, y = generator.next()
    
    # 9개의 이미지를 보여주는 그리드 표시
    plt.figure(figsize=(7, 7))
    if y_pred is None:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            idx = randint(0, len(x) - 1)  # len(x)로 인덱스 범위를 제한
            plt.imshow(x[idx])
            plt.axis("off")
            plt.title("Class:{}".format(labels[np.argmax(y[idx])]))                                                    
    else:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(x[i])
            plt.axis("off")
            plt.title("Actual:{} \nPredicted:{}".format(labels[np.argmax(y[i])], labels[y_pred[i]]))
# 이미지와 라벨을 표시하는 함수 정의

# 학습 이미지 표시
show_images(train_data_gen)

train_data, train_labels = train_data_gen.next()
# 학습 데이터와 라벨 가져오기

# 데이터 형태 출력
print(train_data.shape, train_labels.shape)

sm = SMOTE(random_state=42)
# SMOTE를 사용하여 데이터 불균형 해결

train_data, train_labels = sm.fit_resample(train_data.reshape(-1, IMG_SIZE * IMG_SIZE * 3), train_labels)
# 데이터 재샘플링

train_data = train_data.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
# 데이터 형태 재변경

train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
# 학습, 테스트, 검증 데이터 분할

# 컨볼루션 블록 정의
def conv_block(filters, act='relu'):
    block = Sequential()
    block.add(Conv2D(filters, 3, activation=act, padding='same'))
    block.add(Conv2D(filters, 3, activation=act, padding='same'))
    block.add(BatchNormalization())
    block.add(MaxPool2D())
    return block

# 밀집 블록 정의
def dense_block(units, dropout_rate, act='relu'):
    block = Sequential()
    block.add(Dense(units, activation=act))
    block.add(BatchNormalization())
    block.add(Dropout(dropout_rate))
    return block

IMAGE_SIZE = [128, 128]
act = 'relu'
# 이미지 크기와 활성화 함수 설정

model = Sequential([
    Input(shape=(*IMAGE_SIZE, 3)),
    Conv2D(16, 3, activation=act, padding='same'),
    Conv2D(16, 3, activation=act, padding='same'),
    MaxPool2D(),
    conv_block(32),
    conv_block(64),
    conv_block(128),
    Dropout(0.2),
    conv_block(256),
    Dropout(0.2),
    Flatten(),
    dense_block(512, 0.7),
    dense_block(128, 0.5),
    dense_block(64, 0.3),
    Dense(4, activation='softmax')        
], name="cnn_model")
# CNN 모델 정의

METRICS = [tf.keras.metrics.CategoricalAccuracy(name='acc'),
           tf.keras.metrics.AUC(name='auc'), 
           tfa.metrics.F1Score(num_classes=4)]
# 모델 평가지표 설정

model.compile(optimizer='adam',
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=METRICS)
# 모델 컴파일

model.summary()
# 모델 요약 출력

CALLBACKS = [
    EarlyStopping(monitor='accuracy', min_delta=0.01, patience=5, mode='max'),
    ModelCheckpoint(filepath='C:/Users/RXO/Desktop/daeseong_file/project_code/Dementia/model/model_checkpoint.h5', save_best_only=True)
]
# 콜백 설정 (조기 종료 및 체크포인트 저장)

EPOCHS = 10
# 에포크 설정~

history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=EPOCHS, callbacks=CALLBACKS)
# 모델 학습

test_scores = model.evaluate(test_data, test_labels)
print("Testing Accuracy: %.2f%%"%(test_scores[1] * 100))
# 테스트 데이터로 모델 평가

# 학습 중 메트릭 추세 플로팅
fig, ax = plt.subplots(1, 3, figsize=(30, 5))
ax = ax.ravel()

for i, metric in enumerate(["acc", "auc", "loss"]):
    ax[i].plot(history.history[metric])
    ax[i].plot(history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("Epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

# 최종 모델 저장
model.save('C:/Users/RXO/Desktop/daeseong_file/project_code/Dementia/model/cnn_model.h5')
