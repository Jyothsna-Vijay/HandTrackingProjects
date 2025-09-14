import cv2 
import numpy as np
import time
import HandTrackingModule as htm
import math
import os
import simpleaudio as sa


cap = cv2.VideoCapture(0)
wCam, hCam = 640, 480 #1280, 720
cap.set(3, wCam) # 3 is the width property id
cap.set(4, hCam) # 4 is the height property id
pTime = 0 # Previous time for FPS calculation

detector = htm.handDetector(detectionCon = 0.75)

def playSound():
    os.system("afplay /System/Library/Sounds/Pop.aiff &")

def setVolume(percent):
    percent = max(0, min(100, percent))  # Clamp percent to [0, 100]
    os.system(f"osascript -e 'set volume output volume {percent}'") # macOS command to set volume
    playSound()

vol = 0
volBar = 400
volPercent = 0

while True:
    success, img = cap.read()
    
    img = detector.findHands(img, draw = False) # Find and draw hands
    #img = cv2.flip(img,1)
    lmList = detector.findPosition(img, draw = False) 
    if len(lmList) != 0:
        #print(lmList[4], lmList[8]) # Print the coordinates of the tip of the thumb (id=4) and index finger (id=8)

        x1, y1 = lmList[4][1], lmList[4][2] # Tip of the thumb
        x2, y2 = lmList[8][1], lmList[8][2] # Tip of the index finger

        #cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 # Center point between thumb and index finger
        #cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED) # Draw a filled circle at the tip of the thumb
        #cv2.circle(img, (x2, y2), 7, (255, 0, 255), cv2.FILLED) # Draw a filled circle at the tip of the index finger
        #cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # Draw a line between thumb and index finger
        #cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED) # Draw a filled circle at the center point
        
        length = math.hypot(x2 - x1, y2 - y1) # Calculate the distance between thumb and index finger
        print(length)
        #if length < 50:
            #cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED) # Change center circle to green if fingers are close

        # Hand range 50 - 300
        # Volume range 0 - 100
        # Map length → volume

        vol = np.interp(length, [50, 150], [0, 100]) # Map the length to volume percentage
        print(f'Volume: {int(vol)} %')
        setVolume(int(vol))
        
        volBar = np.interp(length, [50, 150], [400, 150]) # Map the length to volume bar height
        cv2.rectangle(img, (50,150), (85,400), (0,255,0), 3) # Draw the outline of the volume bar
        cv2.rectangle(img, (50,int(volBar)), (85,400), (0,255,0), cv2.FILLED) # Draw the filled volume bar
        cv2.putText(img, f'{int(vol)} %', (40,450), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3) # Display volume percentage
    
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3) # Display FPS on the image
       

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()