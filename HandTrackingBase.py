import cv2
import mediapipe as mp
from mediapipe.tasks import python
import math
import numpy as np
from mediapipe.tasks.python import vision

class handDetector():
    def __init__(self, maxHands=2, detectionCon=0.5, trackCon=0.7):
        self.latest_result = None
        self.tipIDs = [4, 8, 12, 16, 20]

        def _callback(result, output_image, timestamp_ms):
            self.latest_result = result

        options =  mp.tasks.vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=maxHands,                    # these belong inside HandLandmarkerOptions,
            min_hand_presence_confidence=detectionCon,
            min_tracking_confidence=trackCon, 
            result_callback=_callback)
        
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def findHands(self, frame, draw = True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)  #new API requires mp.Image
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)  # detect_async requires a timestamp
        self.landmarker.detect_async(mp_image, timestamp_ms)  #replaces hands.process(), results go to callback

        if draw and self.latest_result and self.latest_result.hand_landmarks:
            h, w, _ = frame.shape                                  
            for hand in self.latest_result.hand_landmarks:                   
                for lm in hand:                                           
                    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 5, (0,255,0), -1) 
                #connections on hand
                for a, b in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),   
                            (7,8),(0,9),(9,10),(10,11),(11,12),(0,13),     
                            (13,14),(14,15),(15,16),(0,17),(17,18),        
                            (18,19),(19,20)]:                             
                        cv2.line(frame,(int(hand[a].x*w),int(hand[a].y*h)),   
                            (int(hand[b].x*w),int(hand[b].y*h)),(255,0,0),2)  
        return frame

    def findPosition(self, frame, handNum=0, draw=True):
        self.lmList = []
        bbox = []
        if self.latest_result and self.latest_result.hand_landmarks:
            h, w, c = frame.shape
            hand = self.latest_result.hand_landmarks[handNum]
            xList, yList = [], []
            for id, lm in enumerate(hand):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cz = lm.z
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(frame, (xmin-20, ymin-20), (xmax+20, ymax-20), (0, 255, 0), 2)
        return self.lmList, bbox

    def fingersUp(self, lmList):
        fingers = []
        if len(lmList) == 0:
            return fingers
        #must change to >
        fingers.append(1 if lmList[self.tipIDs[0]][1] < lmList[self.tipIDs[0]-1][1] else 0)
        for id in range(1, 5):
            fingers.append(1 if lmList[self.tipIDs[id]][2] < lmList[self.tipIDs[id]-2][2] else 0)
        return fingers 

    def findDistance(self, p1, p2, lmList, frame, draw=True):
        x1, y1 = lmList[p1][1], lmList[p1][2]
        x2, y2 = lmList[p2][1], lmList[p2][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        if draw:
            cv2.line(frame, (x1,y1), (x2, y2), (255,0,255), 3)
            cv2.circle(frame, (x1,y1), 15, (255,0,255), cv2.FILLED)
            cv2.circle(frame, (x2,y2), 15, (255,0,255), cv2.FILLED)
            cv2.circle(frame, (cx,cy), 15, (255,0,255), cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)
        return length, frame, [x1, y1, x2, y2, cx, cy]
    
    def findDepth(self, lmList):
        if len(lmList) == 0:
            return 0
        #put landmarks on points 5 & 7 for palm width estimation
        x1, y1 = lmList[5][1], lmList[5][2]
        x2, y2 = lmList[17][1], lmList[17][2]
        #pixel distance
        px_dist = math.hypot(x2-x1, y2-y1)
        return px_dist



def main():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FPS, 24)
    detector = handDetector()

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = detector.findHands(frame)
        lmList, bbox = detector.findPosition(frame)

        cv2.imshow("Hands", frame)
        if cv2.waitKey(1) & 0xFF == 27: #press esc to exit
             break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()