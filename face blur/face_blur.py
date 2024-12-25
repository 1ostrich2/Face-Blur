import cv2
import numpy as np

video = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    r, frame = video.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #color change
    
    face = cascade.detectMultiScale(frame, 1.05, 4)
    
    for x,y,w,h in face:
        roi = frame[x:x+w, y:y+h]
        roi = cv2.medianBlur(roi, 75)
        frame[x:x+w, y:y+h] = roi
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,0), 2)


    cv2.imshow('video', frame)
    t=cv2.waitKey(1)

    if t == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
