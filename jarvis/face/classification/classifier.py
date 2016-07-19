import pickle

import cv2
import numpy as np
import os

from jarvis.face.classification.torch_neural_net import TorchNeuralNet
from jarvis.face.preprocessing.aligndlib import AlignDlib


def get_features(image_path):
    # TODO: Maybe split this function in two. (Care with efficiency)

    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise AttributeError("Unable to load image: {}".format(bgr_image))

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    print "  + Original size: {}".format(rgb_image.shape)

    align = AlignDlib('../models/dlib/shape_predictor_68_face_landmarks.dat')
    bounding_box = align.getLargestFaceBoundingBox(rgb_image)
    if bounding_box is None:
        raise Exception("Unable to find a face: {}".format(image_path))

    # TODO: Parametrize size
    aligned_face = align.align(96, rgb_image, bounding_box, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    if aligned_face is None:
        raise Exception("Unable to align image: {}".format(image_path))

    net = TorchNeuralNet('/Users/albertocastano/development/jarvis/jarvis/face/models/openface/nn4.small2.v1.t7',
                         imgDim=96, cuda=False)
    rep = net.forward(aligned_face)
    return rep


def main():
    images = ['/Users/albertocastano/development/lfw_funneled/Alberto_Castano/Alberto_Castano_0001.jpg']

    model_file = '/Users/albertocastano/development/features/classifier.pkl'
    with open(model_file, 'r') as f:
        print "Loading model from {}".format(model_file)
        (label_encoder, clf) = pickle.load(f)

    for img in images:
        print "Processing: {}".format(img)
        features = get_features(img)
        predictions = clf.predict_proba(features).ravel()
        max_I = np.argmax(predictions)
        person_name = label_encoder.inverse_transform(max_I)
        confidence = predictions[max_I]

        print "Predicted {} with {:.2f} confidence".format(person_name, confidence)


if __name__ == '__main__':
    main()
