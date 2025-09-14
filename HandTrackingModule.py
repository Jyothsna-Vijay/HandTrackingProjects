import cv2 
import mediapipe as mp
import time
import math

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode = self.mode, max_num_hands = self.maxHands, min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon) # Default parameters, only uses RGB images
        self.mpDraw = mp.solutions.drawing_utils

        self.tipsIds = [4, 8, 12, 16, 20] # Tips of each finger


    

    def findHands(self, img, draw = True):
        # For webcam 

            # Flip the image horizontally (mirror effect)
            img = cv2.flip(img, 1)  # 1 = horizontal, 0 = vertical, -1 = both
            imgRGB =cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the BGR image to RGB
            self.results = self.hands.process(imgRGB) # Process the RGB image to detect hands
            #print(results.multi_hand_landmarks) # Print the hand landmarks if detected
            
            if self.results.multi_hand_landmarks: # If hands are detected
                for handLms in self.results.multi_hand_landmarks: # For each hand detected
                    if draw:
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) # Draw the hand landmarks + connection on the original BGR image
            return img
    

    def findPosition(self, img, handNo = 0, draw = True): 

        self.lmList = []
        if self.results.multi_hand_landmarks: # If hands are detected
            myHand = self.results.multi_hand_landmarks[handNo] # Get the specified hand
            for id, lm in enumerate(myHand.landmark):# For each index number for each landmark in the hand
                    #print(id, lm) # Print the index number and landmark position (x,y,z)
                    h, w, c = img.shape # Get the height, width, and channels of the original BGR image
                    cx, cy = int(lm.x * w), int(lm.y * h) # Convert the normalized landmark coordinates to pixel coordinates
                    #print(id, cx, cy) # Print the index number and pixel coordinates
                    self.lmList.append([id, cx, cy]) # Add the index number and pixel coordinates to the list
                    #if id == 4 : # If the landmark is the tip of the thumb (id=4)
                    if draw:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED) # Draw a filled circle at the tip of the thumb
        return self.lmList

    def fingersUp(self):
        fingers = []

            # Infer hand type
        if self.lmList[17][1] < self.lmList[5][1]: # If pinky is to the left of index finger
            handLabel = "Right"
        else:
            handLabel = "Left"

        # --- Thumb ---
        if handLabel == "Right":
            if self.lmList[self.tipsIds[0]][1] < self.lmList[self.tipsIds[0]-1][1]: # If tip of thumb is to the left of the joint
                fingers.append(0)
            else:
                fingers.append(1)
        else:  # Left hand
            if self.lmList[self.tipsIds[0]][1] > self.lmList[self.tipsIds[0]-1][1]: # If tip of thumb is to the right of the joint
                fingers.append(0)
            else:
                fingers.append(1)
            
        # 4 Fingers
        for id in range(1,5):
            if self.lmList[self.tipsIds[id]][2] < self.lmList[self.tipsIds[id] - 2][2]: # If tip of index finger is above the middle joint
                fingers.append(1) # Finger is open
            else:
                fingers.append(0)  # Finger is closed
        #print(fingers)
        return fingers
    

    def findDistance(self, p1, p2, img=None):
        x1, y1 = p1
        x2, y2 = p2
        length = math.hypot(x2 - x1, y2 - y1)

        if img is not None:
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED) # Draw a filled circle at the first point
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED) # Draw a filled circle at the second point
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED) # Draw a filled circle at the center point

        return length
                    
   
    
# Dummy code 
def main():
    cap = cv2.VideoCapture(0) 
    pTime = 0 # Previous time for FPS calculation
    cTime = 0 # Current time for FPS calculation
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img) # Find and draw hands
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[12])
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3) # Display FPS on the image
       

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()





if __name__ == "__main__":
    main()