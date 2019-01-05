# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 11:03:29 2018

@author: Paweł
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Activation, Dropout
from keras import backend as Keras_backend
import cv2
import numpy as np
import time
import pafy
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image

img_width, img_height = 150, 150

if Keras_backend.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

model.load_weights('powazny.h5')

#mypath = '..\\..\\..\\neutral_network\\dataset\\data_validation\\nonfire\\'
#mypath2 = '..\\..\\..\\neutral_network\\dataset\\data_validation\\Images\\n02085936-Maltese_dog\\'
#onlyfiles = [f for f in listdir(mypath2) if isfile(join(mypath2, f))]
#fire_counter = 0
#not_fire_counter = 0
#for f in onlyfiles:
#    test_image = image.load_img(mypath2 + f, target_size = (img_width,img_height))
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
#print('All images: ' + str(fire_counter + not_fire_counter) + ' Fire detected on ' + str(fire_counter) + 'Fire detected on ' + str(100*fire_counter/(fire_counter + not_fire_counter)) + ' %')

video = cv2.VideoCapture("../../flame2.mp4")

# odkomentować w celu testów pliku wideo na dysku
#video = cv2.VideoCapture(video)
# odkomentować w celu testów z kamerki w laptopie
#video = cv2.VideoCapture(0)
url = "https://www.youtube.com/watch?v=BoQUalGOGWk"
videoPafy = pafy.new(url)
best = videoPafy.getbest(preftype="webm")
video = cv2.VideoCapture(best.url)

#video = cv2.VideoCapture(0)
while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break
    test_image = cv2.resize(frame, (img_width, img_height))
    #test_image = image.load_img(mypath + f, target_size = (img_width,img_height))
    test_image = image.img_to_array(test_image)/255
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    if result[0][0] <= 0.5:
        predict = 'fire ' + str(100 - round(result[0][0] * 100,3)) + '%'
        print(str(result[0][0]))
        #not_fire_counter +=1
    else:
        predict = 'not fire '
        print(str(result[0][0]))
        #fire_counter +=1
    cv2.putText(frame, predict,(100,100), cv2.FONT_HERSHEY_SIMPLEX, 3,(0,255,255),4,cv2.LINE_AA)
    cv2.imshow("output", frame)
    #time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()