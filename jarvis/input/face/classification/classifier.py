import pickle

import cv2
import numpy as np
from jarvis.face.classification.torch_neural_net import TorchNeuralNet
from jarvis.face.utils import get_config

from jarvis.input.face.preprocessing.aligndlib import AlignDlib
from jarvis.log import init_logger


class Classifier:
    def __init__(self):
        self.logger = init_logger(self.__class__.__name__)

    def _preprocess_and_get_features(self, image_path, aligner, net):
        bgr_image = cv2.imread(image_path)
        if bgr_image is None:
            raise AttributeError("Unable to load image: {}".format(bgr_image))

        aligned_face = self._preprocess(bgr_image, aligner)
        features = self._get_features(aligned_face, net)
        return features

    def _preprocess(self, bgr_image, aligner):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        self.logger.info("  + Original size: {}".format(rgb_image.shape))

        bounding_box = aligner.getLargestFaceBoundingBox(rgb_image)
        if bounding_box is None:
            raise Exception("Unable to find a face")

        img_dim = int(get_config().get('PreProcessing', 'FaceSize'))
        aligned_face = aligner.align(img_dim, rgb_image, bounding_box, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
        if aligned_face is None:
            raise Exception("Unable to align image")

        return aligned_face

    def _get_features(self, face, net):
        rep = net.forward(face)
        return rep

    def predict(self):
        images = []
        for i in range(1, 29):
            if i == 17:
                continue
            number = "%02d" % i
            images.append(

                '/Users/albertocastano/development/lfw_funneled/Alberto_Castano/Alberto_Castano_00{}.jpg'.format(number))

        model_file = '/Users/albertocastano/development/features/classifier.pkl'
        with open(model_file, 'r') as f:
            self.logger.info("Loading model from {}".format(model_file))
            (label_encoder, clf) = pickle.load(f)

        aligner = AlignDlib('../models/dlib/shape_predictor_68_face_landmarks.dat')
        net = TorchNeuralNet('/Users/albertocastano/development/jarvis/jarvis/face/models/openface/nn4.small2.v1.t7',
                             imgDim=96, cuda=False)

        for img in images:
            self.logger.info("Processing: {}".format(img))

            features = self._preprocess_and_get_features(img, aligner, net).reshape(1, -1)
            predictions = clf.predict_proba(features).ravel()
            max_I = np.argmax(predictions)
            person_name = label_encoder.inverse_transform(max_I)
            confidence = predictions[max_I]

            self.logger.info("  + Predicted {} with {:.2f} confidence".format(person_name, confidence))
