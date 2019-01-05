#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Fri Dec 28 11:11:34 2018

@author: Paweł
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications, Input

img_width, img_height = 150, 150

top_model_weights_path = 'VGG16modlel.h5'
train_data_dir = 'dataset/data_train'
validation_data_dir = 'dataset/data_validation'
nb_train_samples = 1264
nb_validation_samples = 608
epochs = 50
batch_size = 16
input_shape = (img_width, img_height, 3)

def create_data_to_match_VGG16():
    data_generator = ImageDataGenerator(rescale=1. / 255)

    format_input_image = Input(shape=(150,150,3))
    model = applications.VGG16(include_top=False, weights='imagenet', input_tensor=format_input_image)

    generator = data_generator.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    created_trening_data = model.predict_generator(generator, nb_train_samples // batch_size)
    np.save(open('VGG16modlel_train.npy', 'wb'),created_trening_data)

    generator = data_generator.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    created_validation_data = model.predict_generator(generator, nb_validation_samples // batch_size)
    np.save(open('VGG16modlel_validation.npy', 'wb'),created_validation_data)


def train_pre_trained_VGG16_model():
    train_data = np.load(open('VGG16modlel_train.npy', "rb"))
    train_labels = np.array([0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))

    validation_data = np.load(open('VGG16modlel_validation.npy',"rb"))
    validation_labels = np.array([0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


create_data_to_match_VGG16()
train_pre_trained_VGG16_model()