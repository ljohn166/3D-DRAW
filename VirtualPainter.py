import cv2
import numpy as np
import time
import os
#import HandTrackingBase as htb

folderPath = "header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FPS, 24)
capture.set(3, 1920)
capture.set(4, 720)

while True:
    success, image = capture.read()
    
    image[0:120,0:1920] = header
    cv2.imshow("icons",image)
    cv2.waitKey(1)

    
    


