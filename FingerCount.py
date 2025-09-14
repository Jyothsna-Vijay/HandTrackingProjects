#Maybe ass pictures 

import cv2
import time 
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

#folderPath = "FingerImages"
#myList = os.listdir(folderPath)
#myList =  [f for f in myList if f.endswith(".jpg")]
#myList.sort()
#print(myList)

#overlayList = []
#for imPath in myList:
#    image = cv2.imread(f'{folderPath}/{imPath}') # read image
#    overlayList.append(image)

#print(len(overlayList))

pTime = 0 # Previous time for FPS calculation

detector = htm.handDetector(detectionCon=0.75)

tipsIds = [4, 8, 12, 16, 20] # Tips of each finger

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera")
        break

    img = detector.findHands(img, draw = False) # Hand Mapping
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)
    
    if len(lmList) != 0:
        fingers = []

            # Infer hand type
        if lmList[17][1] < lmList[5][1]: # If pinky is to the left of index finger
            handLabel = "Right"
        else:
            handLabel = "Left"

        # --- Thumb ---
        if handLabel == "Right":
            if lmList[tipsIds[0]][1] < lmList[tipsIds[0]-1][1]: # If tip of thumb is to the left of the joint
                fingers.append(0)
            else:
                fingers.append(1)
        else:  # Left hand
            if lmList[tipsIds[0]][1] > lmList[tipsIds[0]-1][1]: # If tip of thumb is to the right of the joint
                fingers.append(0)
            else:
                fingers.append(1)
            
        # 4 Fingers
        for id in range(1,5):
            if lmList[tipsIds[id]][2] < lmList[tipsIds[id] - 2][2]: # If tip of index finger is above the middle joint
                fingers.append(1) # Finger is open
            else:
                fingers.append(0)  # Finger is closed
        #print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)
        cv2.putText(img, str(totalFingers), (43,375), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,255), 10) # Display FPS on the image

    #overlay = overlayList[0]
    #h, w, c = overlayList.shape

    #img[0:h, 0:w] = overlayList[2] #Come back to this


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3) # Display FPS on the image
    
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()