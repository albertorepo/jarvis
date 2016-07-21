import os
import pickle
from operator import itemgetter

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from jarvis.face.utils import get_config
from jarvis.log import init_logger


class Trainer:
    def __init__(self):
        self.logger = init_logger(self.__class__.__name__)

    def train(self):
        input_dir = get_config().get('Classification', 'TrainingInputPath')
        self.logger.info("Loading features")
        file_name = os.path.join(input_dir, 'labels.csv')
        labels = pd.read_csv(file_name, header=None).as_matrix()[:, 1]
        labels = map(itemgetter(1),
                     map(os.path.split,
                         map(os.path.dirname, labels)))
        label_encoder = LabelEncoder().fit(labels)
        labels_encoded = label_encoder.transform(labels)
        num_classes = len(label_encoder.classes_)

        file_name = os.path.join(input_dir, 'reps.csv')
        features = pd.read_csv(file_name, header=None).as_matrix()

        self.logger.info("Training for {} classes.".format(num_classes))

        clf = SVC(C=1, kernel='linear', probability=True)

        # TODO: Try a previous LDA
        clf.fit(features, labels_encoded)

        file_name = os.path.join(input_dir, 'classifier.pkl')
        self.logger.info("Saving classifier to '{}'".format(file_name))
        with open(file_name, 'w') as f:
            pickle.dump((label_encoder, clf), f)
