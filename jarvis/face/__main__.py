from argparse import ArgumentParser
from time import time

import cv2
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from jarvis.face.imageloader import ImageLoader, Bunch


def load_images_from_path(path):
    return ImageLoader().load_image_from_path(path, min_images_per_folder=15, max_images_per_folder=15)


def preprocessing(images, casc_path='cascade.xml'):
    faces_data = []
    target = images.target
    face_cascade = cv2.CascadeClassifier(casc_path)
    for (ind, img) in enumerate(images.images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        try:
            (x, y, w, h) = faces[0]
            area = w * h
            for (xt, yt, wt, ht) in faces:
                if wt * ht > area:
                    (x, y, w, h) = (xt, yt, wt, ht)

        except:
            print "WARNING - Face not detected"
            target[ind] = -1
            continue

        face = gray[y:y + h, x:x + w]
        faces_data.append(face)

    faces_data = faces_data

    return Bunch(data=faces_data, images=faces_data,
                 target=images.target[np.where(images.target != -1)],
                 target_names=images.target_names)


def main():
    # TODO: Re-write this
    args_parser = ArgumentParser(prog='__main__', description='Face recognition')
    args_parser.add_argument('-p', dest="img_path", help="Path to the images.")
    args = args_parser.parse_args()

    images = load_images_from_path(args.img_path)
    images = preprocessing(images)
    X = images.images

    y = images.target
    target_names = images.target_names
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
            if images.target_names[face_predicted] == 'Alberto Castano':
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
