# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 10:48:16 2018

@author: Paweł
"""

import numpy as np
import cv2
import os

min_color_values = np.array([0, 120, 120], dtype="uint8")
max_color_values = np.array([45, 220, 255], dtype="uint8")

path = '../../przygotowaniedanych/fire/'
name_list = os.listdir(path)
dataset_size = len(name_list)
font = cv2.FONT_HERSHEY_SIMPLEX

def read_img(path):
    if os.path.isfile(path):
        image = cv2.imread(path)
        if image is not None:
            return image
        else:
            raise ValueError('Error during opening image with path: {}'.format(path))
    else:
        raise ValueError('Path provided is not a valid file: {}'.format(path))

fire_counter = 0

for name in name_list:
    image = read_img(path + name)  
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, min_color_values, max_color_values)

    (image2, countours, hierarchy) = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  

    fire = 'not fire'

    for contour_field in countours:
        if cv2.contourArea(contour_field) < 500:
            continue
        fire = 'fire'
#        (x, y, w, h) = cv2.boundingRect(contour_field)
#        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#        rect = cv2.minAreaRect(contour_field)  
#        box = np.int0(cv2.boxPoints(rect))  
#        box = np.int0(box)  
#        cv2.drawContours(image, [box], 0, (0, 191, 255), 2)  
#        cv2.imshow('Color', image) 
    print(name + " " + str(fire))
    if fire == 'fire':
        fire_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'): break
print('All images: ' + str(dataset_size) + ' Fire detected on ' + str(fire_counter) + 'Fire detected on ' + str(100*fire_counter/dataset_size) + ' %')
#cv2.destroyAllWindows()