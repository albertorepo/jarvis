import argparse

from jarvis.face.classification.classifier import Classifier
from jarvis.face.classification.training import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train or use the classifier')
    parser.add_argument('mode', type=str, help='The classification mode [train/predict/realtime]', default='realtime',
                        choices=['train', 'predict', 'realtime'])

    args = parser.parse_args()

    if args.mode == 'train':
        trainer = Trainer()
        trainer.train()
    elif args.mode == 'predict':
        classifier = Classifier()
        classifier.predict()
    elif args.mode == 'realtime':
        raise NotImplementedError("Not available for the moment, built it motherfucker")


if __name__ == '__main__':
    main()
