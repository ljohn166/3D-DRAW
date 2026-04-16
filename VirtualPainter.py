import cv2
import numpy as np
import time
import os
import random
import HandTrackingBase as htb

folderPath = "header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]

#random.randint(1, 255)
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FPS, 24)
capture.set(3, 1920)
capture.set(4, 720)

detector = htb.handDetector(trackCon=0.75, maxHands=2)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, image = capture.read()
    image = cv2.flip(image, 1);
    #landmarks
    image = detector.findHands(image)

    
    image[0:120,0:1920] = header
    cv2.imshow("icons",image)
    cv2.waitKey(1)

    
    


