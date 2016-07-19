import os
import pickle
from operator import itemgetter

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def main():
    # TODO: Parametrize this
    # TODO: Paths as os.join
    input_dir = '/Users/albertocastano/development/features'
    print "Loading features"
    file_name = "{}/labels.csv".format(input_dir)
    labels = pd.read_csv(file_name, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))
    label_encoder = LabelEncoder().fit(labels)
    labels_encoded = label_encoder.transform(labels)
    num_classes = len(label_encoder.classes_)

    file_name = "{}/reps.csv".format(input_dir)
    features = pd.read_csv(file_name, header=None).as_matrix()

    print "Training for {} classes.".format(num_classes)

    clf = SVC(C=1, kernel='linear', probability=True)

    # TODO: Try a previous LDA
    clf.fit(features, labels_encoded)

    file_name = "{}/classifier.pkl".format(input_dir)
    print "Saving classifier to '{}'".format(file_name)
    with open(file_name, 'w') as f:
        pickle.dump((label_encoder, clf), f)


if __name__ == '__main__':
    main()