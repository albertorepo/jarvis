import os
from argparse import ArgumentParser
from time import time

import cv2
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from jarvis.face.imageloader import ImageLoader, Bunch



def main():
    # TODO: Re-write this
    args_parser = ArgumentParser(prog='__main__', description='Face recognition')
    args_parser.add_argument('-p', dest="img_path", help="Path to the images.")
    args = args_parser.parse_args()

    image_loader = ImageLoader()
    image_loader.load_images_from_path(args.img_path, min_images_per_folder=10, max_images_per_folder=15)
    image_loader.preprocessing('cascade.xml')
    X = image_loader.data.images

    y = image_loader.data.target == np.where(image_loader.data.target_names == 'Alberto Castano')[0]
    target_names = image_loader.data.target_names
    n_classes = target_names.shape[0]
    print("Total dataset size:")
    print("n_classes: %d" % n_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    recognizer = cv2.face.createLBPHFaceRecognizer()
    recognizer.train(list(X_train), y_train)

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = []
    for img in list(X_test):
        pred = recognizer.predict(img)
        y_pred.append(pred)
    y_pred = np.asarray(y_pred)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

    cascPath = 'cascade.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)
    video_capture = cv2.VideoCapture(0)
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
            face_predicted = recognizer.predict(face)
            if image_loader.data.target_names[face_predicted] == 'Alberto Castano':
                print "Bienvenido, amo"
                print ""
                exit()

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
