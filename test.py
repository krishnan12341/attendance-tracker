from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

video = cv2.VideoCapture(0)
faceDetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')


with open('data/names.pkl', 'rb') as f:
    LABELS=pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)

#1 name label vs 100 diff pictures, match each label to each frame so code works
if len(LABELS) != len(FACES):
    LABELS = np.arange(len(FACES) - 1) # -1 to account for inputted name

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)


imgBackground = cv2.imread("backdrop.png")

colNames = ['NAME', 'TIME']


while True:
    ret,frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    for(x,y,w,h) in faces:
        cropImg = frame[y:y+h, x:x+w, :]
        resize = cv2.resize(cropImg, (50,50)).flatten().reshape(1, -1)
        knnOut = knn.predict(resize)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

        #background rectangle
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)

        cv2.putText(frame, str(knnOut[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 3)

        attendance = [str(knnOut[0]), str(timestamp)]


    frame_height, frame_width = frame.shape[:2]

    background_height = 1080
    background_width = 1920
    target_width = 640
    target_height = 480

    aspect_ratio = frame_width / frame_height

    if aspect_ratio > target_width / target_height:
    # Width is the limiting factor
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
    # Height is the limiting factor
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    frame_resized = cv2.resize(frame, (new_width, new_height))
    padded_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    padded_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = frame_resized

    x_position = 625  # Increase this value to move right
    y_position = 310  # Increase this value to move down

    imgBackground[y_position:y_position + target_height, x_position:x_position + target_width] = padded_frame

    cv2.imshow("Frame", imgBackground)
    k = cv2.waitKey(1)

    if k == ord('p'):
        file_path = os.path.join("Attendance", f"Attendance_{date.replace('/', '-')}.csv")
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(colNames)
            writer.writerow(attendance)
            
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()