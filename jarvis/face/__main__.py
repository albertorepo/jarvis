from argparse import ArgumentParser

import cv2
import time

from jarvis.face.imageloader import ImageLoader


def load_images_from_path(path):
    return ImageLoader().load_image_from_path(path, min_images_per_folder=70)


def preprocessing(images):
    # TODO: This is hardcoded here
    casc_path = 'cascade.xml'
    face_cascade = cv2.CascadeClassifier(casc_path)
    for img in images.data:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        cv2.imshow('Video', gray)
        time.sleep(10)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def main():
    # TODO: Re-write this
    args_parser = ArgumentParser(prog='__main__', description='Face recognition')
    args_parser.add_argument('-p', dest="img_path", help="Path to the images.")
    args = args_parser.parse_args()

    images = load_images_from_path(args.img_path)
    preprocessing(images)


if __name__ == '__main__':
    main()