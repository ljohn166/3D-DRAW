import cv2
import numpy as np
import time
import os
import random
from pygame import mixer
import HandTrackingBase as htb

folderPath = "header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imgPath in myList:
    headerImg = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(headerImg)
print(len(overlayList))
header = overlayList[0]
mixer.init()
click = mixer.Sound('click.wav')

#random.randint(1, 255)
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FPS, 24)
capture.set(3, 1920)
capture.set(4, 720)

detector = htb.handDetector(trackCon=0.75, maxHands=2)
xp, yp = 0, 0
imgCanvas = None

#temp random color
rVal = random.randint(1, 255)
gVal = random.randint(1, 255)
bVal = random.randint(1, 255)
prev_selection = 0
brush_size = 12
big_brush = False

while True:
    success, image = capture.read()
    image = cv2.flip(image, 1)
    if imgCanvas is None:
        imgCanvas = np.zeros_like(image)
    #landmarks
    image = detector.findHands(image)
    lmList, _ = detector.findPosition(image, draw=False)


    if len(lmList) >= 21:
        #tip of index & mid finger
        x1, y1 = lmList[8][1], lmList[8][2]
        x2, y2 = lmList[12][1], lmList[12][2]
        #check which finger is up
        fingers = detector.fingersUp(lmList)

        print(fingers)
        #determine if in drawing mode or selection mode
        if fingers[1]==1 and fingers[2]==1:
            xp, yp = 0, 0
            cv2.rectangle(image, (x1, y1-brush_size), (x2, y2+brush_size), (rVal, gVal, bVal), cv2.FILLED)
            print("Selection Mode")
            if y1 < 120:
                current_selection = 0
                #check for click
                if 320<x1<500:
                    current_selection=1
                elif 875<x1<1040:
                    current_selection=2
                elif 1410<x1<1580:
                    current_selection=3

                if current_selection != 0 and current_selection != prev_selection:
                    click.play()
                

                if current_selection == 3:
                    rVal = random.randint(1, 255)
                    gVal = random.randint(1, 255)
                    bVal = random.randint(1, 255)
                    if big_brush:
                        brush_size = 23
                    else:
                        brush_size = 12

                elif current_selection == 2:
                    rVal, gVal, bVal = 0,0,0
                    brush_size = 60

                elif current_selection == 1:
                    #change brush size
                    big_brush = not big_brush
                    if big_brush:
                        brush_size = 23
                    else:
                        brush_size = 12
                        
                prev_selection = current_selection

            else:
                prev_selection = 0

        if fingers [1]==1 and fingers[2] == 0:
            cv2.circle(image, (x1, y1), brush_size, (rVal, gVal, bVal), cv2.FILLED)
            print("Drawing Mode")
            if xp==0 and yp==0:
                xp, yp = x1, y1
            cv2.line(image, (xp, yp), (x1, y1), (rVal, gVal, bVal), brush_size)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), (rVal, gVal, bVal), brush_size)
            xp = x1
            yp = y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInverse = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR)
    image = cv2.bitwise_and(image, imgInverse)
    image = cv2.bitwise_or(image, imgCanvas)

    
    image[0:120,0:1920] = header
    #image = cv2.addWeighted(image, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("icons",image)
    cv2.imshow("canvas", imgCanvas)
    cv2.waitKey(1)