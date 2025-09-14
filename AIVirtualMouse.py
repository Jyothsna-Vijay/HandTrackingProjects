import cv2 
import numpy as np
import time 
import os 
import HandTrackingModule as htm
import autopy

wCam, hCam = 640, 480 
frameR = 50 # Frame Reduction
smooth = 5
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, 640) # Width
cap.set(4, 480) # Height

pTime = 0
detector = htm.handDetector( maxHands=1) #detectionCon=0.85,
wScr, hScr = autopy.screen.size() # Get the size of the screen
#print(wScr, hScr)

while True:
    #1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img) # Hand Mapping
    lmList = detector.findPosition(img, draw=False)
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

    #2. Get tip of index + middle
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:] # Index Finger
        x2, y2 = lmList[12][1:] # Middle Finger
        #print(x1, y1, x2, y2)
  
        #3. Check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)

    
    #4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:

            #print("Moving Mode")
        #5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            
            #6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smooth
            clocY = plocY + (y3 - plocY) / smooth

            #7. Move Mouse
            autopy.mouse.move(clocX, clocY)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

    #8. Both Index and middle fingers are up : Selection Mode
        if fingers[1] == 1 and fingers[2] == 1:
            #9. Find distance between fingers
            length = detector.findDistance((x1, y1), (x2, y2), img) # Find distance between index and middle finger
            print(length)
            if length < 25:
                cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED) 
                #10. Click mouse if distance short
                autopy.mouse.click()
            #print("Selection Mode")
        
        

    #11. Frame Rate

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    #cv2.putText(img, f'FPS: {int(fps)}', (400,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3) # Display FPS on the image
    
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()