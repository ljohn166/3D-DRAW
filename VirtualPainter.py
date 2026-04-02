import cv2
import numpy as np
import time
import os
import HandTrackingBase

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

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)  #new API requires mp.Image
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)  # detect_async requires a timestamp
        landmarker.detect_async(mp_image, timestamp_ms)  #replaces hands.process(), results go to callback

        if latest_result:
            h, w, _ = frame.shape                                      
            for hand in latest_result.hand_landmarks:                   
                for lm in hand:                                           
                    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 5, (0,255,0), -1) 
                for a, b in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),   
                             (7,8),(0,9),(9,10),(10,11),(11,12),(0,13),     
                             (13,14),(14,15),(15,16),(0,17),(17,18),        
                             (18,19),(19,20)]:                             
                    cv2.line(frame,(int(hand[a].x*w),int(hand[a].y*h)),   
                             (int(hand[b].x*w),int(hand[b].y*h)),(255,0,0),2)  

        cv2.imshow("Hands", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

capture.release()
cv2.destroyAllWindows()


