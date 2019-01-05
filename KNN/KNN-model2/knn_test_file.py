# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 11:11:34 2018

@author: Paweł
"""

from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense, Dropout
import cv2
import numpy as np
from keras import optimizers
from keras import applications
from keras.models import Model
from keras.preprocessing import image
import pafy
import time
from os import listdir
from os.path import isfile, join

img_width, img_height = 150, 150
top_model_weights_path = 'VGG16modlel.h5'
train_data_dir = 'dataset/data_train'
validation_data_dir = 'dataset/data_validation'
nb_train_samples = 1264
nb_validation_samples = 608
epochs = 50
batch_size = 16

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
print('Model loaded.')

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

top_model.load_weights(top_model_weights_path)

model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

for layer in model.layers[:25]:
    layer.trainable = False

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

#model.save_weights('wytrenowanaVGG.h5')
model.load_weights('wytrenowanaVGG.h5')

fire_counter = 0
not_fire_counter = 0

#mypath = '..\\..\\..\\neutral_network\\dataset\\data_validation\\nonfire\\'
#mypath2 = '..\\..\\..\\neutral_network\\dataset\\data_validation\\Images\\n02085620-Chihuahua\\'
#
#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#for f in onlyfiles:
#    test_image = image.load_img(mypath + f, target_size = (img_width,img_height))
#    print(f)
#    test_image = image.img_to_array(test_image)/255
#    test_image = np.expand_dims(test_image, axis = 0)
#    result = model.predict(test_image)
#    if result[0][0] >= 0.5:
#        predict = 'not_fire'
#        not_fire_counter +=1
#    else:
#        predict = 'fire'
#        fire_counter +=1
#    print(str(result[0][0]))
#print('Fire: ' + str(fire_counter))
#print('Not Fire: ' + str(not_fire_counter))
#print('All images: ' + str(fire_counter + not_fire_counter) + ' Fire detected on ' + str(fire_counter) + ' Fire detected on ' + str(100*fire_counter/(fire_counter + not_fire_counter)) + ' %')

video = "../../flame2.mp4"
# odkomentować w celu testów pliku wideo na dysku
#video = cv2.VideoCapture(video)
# odkomentować w celu testów z kamerki w laptopie
#video = cv2.VideoCapture(0)

url = "https://www.youtube.com/watch?v=BoQUalGOGWk"
videoPafy = pafy.new(url)
best = videoPafy.getbest(preftype="webm")
video = cv2.VideoCapture(best.url)

while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break
    test_image = cv2.resize(frame, (img_width, img_height))
    test_image = image.img_to_array(test_image)/255
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    if result[0][0] <= 0.5:
        predict = 'fire ' + str(100 - round(result[0][0] * 100,3)) + '%'
        print(str(result[0][0]))
        not_fire_counter +=1
    else:
        predict = 'not fire '
        print(str(result[0][0]))
        fire_counter +=1
    cv2.putText(frame, predict,(100,100), cv2.FONT_HERSHEY_SIMPLEX, 3,(0,255,255),4,cv2.LINE_AA)
    cv2.imshow("output", frame)
    #time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()