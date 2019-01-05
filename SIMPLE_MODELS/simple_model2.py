# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 10:42:47 2018

@author: Paweł
"""

import cv2
import numpy as np
import time
import pafy


font = cv2.FONT_HERSHEY_SIMPLEX
video_file = "../flame2.mp4"

# odkomentować w celu testów pliku wideo na dysku
#video = cv2.VideoCapture("../flame2.mp4")
# odkomentować w celu testów z kamerki w laptopie
#video = cv2.VideoCapture(0)
url = "https://www.youtube.com/watch?v=BoQUalGOGWk"
videoPafy = pafy.new(url)
best = videoPafy.getbest(preftype="webm")
video = cv2.VideoCapture(best.url)

def frameDiff(It0, It1, It2):
    dI1 = cv2.absdiff(It2, It1)
    dI2 = cv2.absdiff(It1, It0)
    return cv2.bitwise_and(dI1, dI2)


current_fire_counter = 0
global_fire_counter = 0
previous_img = None
current_img = None
next_img = None

min_color_values = np.array([0, 120, 120], dtype="uint8")
max_color_values = np.array([45, 220, 255], dtype="uint8")

i = 0
while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break
    next_img = frame
    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, min_color_values, max_color_values)

    (im, countours, hierarchy) = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    fire = 'Not Fire'

    for contour_field in countours:
        if cv2.contourArea(contour_field) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour_field)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if previous_img is not None and current_img is not None and next_img is not None:
            if frameDiff(previous_img[y: y + h, x: x + w], current_img[y: y + h, x: x + w], next_img[y: y + h, x: x + w]).sum() > hsv.size * 0.01:
                current_fire_counter += 1
                global_fire_counter += 1
                if current_fire_counter >= 2:
                    rect = cv2.minAreaRect(contour_field)
                    box = np.int0(cv2.boxPoints(rect))
                    box = np.int0(box)
                    cv2.drawContours(frame, [box], 0, (0, 191, 255), 2)
                    fire = 'Fire'
            else:
                current_fire_counter = 0

    cv2.putText(frame, fire, (100, 100), font, 3, (0, 255, 255), 4, cv2.LINE_AA)
    cv2.imshow("output", frame)
    i+=1
    #time.sleep(0.02)
    previous_img = current_img
    current_img = next_img

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
video.release()