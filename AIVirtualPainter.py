import cv2 
import numpy as np
import time 
import os 
import HandTrackingModule as htm

folderPath = "Header"
myList = os.listdir(folderPath)
myList.sort()
print("Files:", myList)

overlayList = []
for imgPath in myList:
    fullPath = f'{folderPath}/{imgPath}'
    image = cv2.imread(fullPath) 
    if image is None:
        print(f"Warning: Could not read image {fullPath}")
    else:
        overlayList.append(image)

print("Loaded overlays: ", len(overlayList))

if len(overlayList) > 0:
    header = overlayList[0]
else:
    raise ValueError("No header images found in the specified folder.")

header = overlayList[0]
drawColor = (97, 105, 225)
brushThickness = 50
rubberThickness = 100
xp, yp = 0, 0
imgCanvas = 255 * np.zeros((720, 1280, 3), np.uint8) # Canvas for drawing h,w,c

#print(os.path.exists("Header/1.jpg"))

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
pTime = 0

detector = htm.handDetector(detectionCon=0.85)

while True:

    # Import image
    success, img = cap.read()
    # Find Hand Landmarks

    img = detector.findHands(img, draw=False) # Hand Mapping
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        
        #print(lmList)
        x1, y1 = lmList[8][1:] # Index finger tip
        x2, y2 = lmList[12][1:] # Middle finger tip

        # Check which fingers are up - draw when one up, select when two are up
        fingers = detector.fingersUp()
        #print(fingers)
        
    # If selection mode - two fingers up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0 # Reset previous points
            print("Selection Mode")
            # Checking for the click
            if y1 < 125:
                if 100 < x1 < 300: # First header - red BGR
                    header = overlayList[0]
                    drawColor = (97, 105, 225)
                elif 350 < x1 < 650: # Second header - pink
                    header = overlayList[1]
                    drawColor = (197, 183, 255)
                elif 675 < x1 < 850: # Third header - orange 
                    header = overlayList[2]
                    drawColor = (122, 160, 255)
                elif 850 < x1 < 1000: # Fourth header - green
                    header = overlayList[3]
                    drawColor = (119, 221, 119)
                elif 1100 < x1 < 1200: # Fourth header - rubber - black
                    header = overlayList[4]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)

            
    # If drawing mode - one finger up, Index finger
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
        
            # If we want to draw continuously, we need to store the previous point
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, rubberThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, rubberThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    imgGrey = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    _, imgInv = cv2.threshold(imgGrey, 50, 255, cv2.THRESH_BINARY_INV) # Create a binary image + Invert it
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR) # Convert back to BGR
    img = cv2.bitwise_and(img, imgInv) # Bitwise AND to remove the drawing from the webcam feed
    img = cv2.bitwise_or(img, imgCanvas) # Bitwise OR to add the drawing back to the webcam feed


    # Resize header to match width if necessary
    header_resized = cv2.resize(header, (1280, 125))
    img[0:125, 0:1280] = header_resized # Overlay the header image on top of the webcam feed
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0) # Merge the canvas and the webcam feed
    #img = cv2.add(img, imgCanvas) # Alternative way to merge the canvas and the webcam feed

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    #cv2.putText(img, f'FPS: {int(fps)}', (400,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3) # Display FPS on the image
    
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    

    