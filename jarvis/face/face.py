import json
import pickle

import cv2
import numpy as np
import time

cascPath = 'cascade.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
model_file = open('model.pkl', 'rb')
clf = pickle.load(model_file)

map_file = open('names_map.json', 'rb')
names_map = json.loads(map_file.read())

model_file.close()
map_file.close()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (168, 168), interpolation=cv2.INTER_CUBIC)
        face = face.reshape(-1).reshape(1, -1)
        face_predicted = clf.predict(face)[0]
        print names_map[str(face_predicted)]
        print ""

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
