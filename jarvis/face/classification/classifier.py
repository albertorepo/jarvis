import pickle

import cv2
import numpy as np

from jarvis.face.classification.torch_neural_net import TorchNeuralNet
from jarvis.face.preprocessing.aligndlib import AlignDlib
from jarvis.face.utils import get_config


def preprocess_and_get_features(image_path, aligner, net):
    aligned_face = preprocess(image_path, aligner)
    features = get_features(aligned_face, net)
    return features


def preprocess(image_path, aligner):
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise AttributeError("Unable to load image: {}".format(bgr_image))

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    print "  + Original size: {}".format(rgb_image.shape)

    bounding_box = aligner.getLargestFaceBoundingBox(rgb_image)
    if bounding_box is None:
        raise Exception("Unable to find a face: {}".format(image_path))

    img_dim = int(get_config().get('Preprocessing', 'FaceSize'))
    aligned_face = aligner.align(img_dim, rgb_image, bounding_box, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    if aligned_face is None:
        raise Exception("Unable to align image: {}".format(image_path))

    return aligned_face


def get_features(face, net):
    rep = net.forward(face)
    return rep


def predict():
    images = ['/Users/albertocastano/development/lfw_funneled/George_W_Bush/George_W_Bush_0087.jpg']

    model_file = '/Users/albertocastano/development/features/classifier.pkl'
    with open(model_file, 'r') as f:
        print "Loading model from {}".format(model_file)
        (label_encoder, clf) = pickle.load(f)

    aligner = AlignDlib('../models/dlib/shape_predictor_68_face_landmarks.dat')
    net = TorchNeuralNet('/Users/albertocastano/development/jarvis/jarvis/face/models/openface/nn4.small2.v1.t7',
                         imgDim=96, cuda=False)

    for img in images:
        print "Processing: {}".format(img)

        features = preprocess_and_get_features(img, aligner, net).reshape(1, -1)
        predictions = clf.predict_proba(features).ravel()
        max_I = np.argmax(predictions)
        person_name = label_encoder.inverse_transform(max_I)
        confidence = predictions[max_I]

        print "Predicted {} with {:.2f} confidence".format(person_name, confidence)
