# IMPORTING ALL THE NECESSARY LIBRARIES
import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np
  
#MODEL INITIALIZATION
mpHands=mp.solutions.hands
hands=mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)
Draw=mp.solutions.drawing_utils
  
#WEBCAM INITIALIZATION
cap=cv2.VideoCapture(0)


#READING IMAGE IN THE FRAME AND PROCESSING THE RGB IMAGE
while True:
    success,frame=cap.read()
    frame=cv2.flip(frame,1)
    frameRGB=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Process=hands.process(frameRGB)
    
    #DETECTING HANDMARKS
    landmarkList=[]
    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:
            for id,landmarks in enumerate(handlm.landmark):
                height,width,color_channels=frame.shape                        # height and width setting
                x,y=int(landmarks.x*width),int(landmarks.y*height)             #calculating the x,y co ordinates
                landmarkList.append([id,x,y])
            Draw.draw_landmarks(frame, handlm,
                                mpHands.HAND_CONNECTIONS)
  
    if landmarkList!=[]:
       
        x1,y1=landmarkList[4][1], landmarkList[4][2]                          #x,y co ordinates for the tip of thumb and index finger
        x2,y2=landmarkList[8][1], landmarkList[8][2]
  
        cv2.circle(frame, (x1, y1), 7, (0, 255, 0), cv2.FILLED)               #drawing circle and line to the tip of thumb and index finger
        cv2.circle(frame, (x2, y2), 7, (0, 255, 0), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        L=hypot(x2-x1,y2-y1)                                                  #calculating square root of sum of squares to specified arguments
  
       
        b_level=np.interp(L,[15, 220],[0, 100])                                # 1-D linear interpolant to a functionwith given discrete 
        sbc.set_brightness(int(b_level))                                       #data points(Hand range 15 - 220, Brightnessrange 0 - 100)
  

    cv2.imshow('Image',frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyAllWindows