import pickle

import cv2
import numpy as np
from jarvis.face.classification.torch_neural_net import TorchNeuralNet
from jarvis.face.utils import get_config

from jarvis.input.face.preprocessing.aligndlib import AlignDlib


def _preprocess_and_get_features(bgr_image, aligner, net):
    if bgr_image is None:
        raise AttributeError("Unable to load image: {}".format(bgr_image))

    aligned_face = _preprocess(bgr_image, aligner)
    if aligned_face is None:
        return None
    features = _get_features(aligned_face, net)
    return features


def _preprocess(bgr_image, aligner):
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    bounding_box = aligner.getLargestFaceBoundingBox(rgb_image)
    if bounding_box is None:
        return None

    img_dim = int(get_config().get('PreProcessing', 'FaceSize'))
    aligned_face = aligner.align(img_dim, rgb_image, bounding_box, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    if aligned_face is None:
        raise Exception("Unable to align image")

    return aligned_face


def _get_features(face, net):
    rep = net.forward(face)
    return rep


def main():

    model_file = get_config().get("Classification", "ModelPath")
    with open(model_file, 'r') as f:
        print "Loading model from {}".format(model_file)
        (label_encoder, clf) = pickle.load(f)

    aligner = AlignDlib(get_config().get("Classification", "AlignerPAth"))
    net = TorchNeuralNet(get_config().get("Classification", "NetPAth"),
                         imgDim=96, cuda=False)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        features = _preprocess_and_get_features(frame, aligner, net)
        if features is not None:
            features = features.reshape(1, -1)
            predictions = clf.predict_proba(features).ravel()
            max_I = np.argmax(predictions)
            person_name = label_encoder.inverse_transform(max_I)
            confidence = predictions[max_I]

            print "  + Predicted {} with {:.2f} confidence".format(person_name, confidence)

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
