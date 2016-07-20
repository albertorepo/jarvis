import argparse

from jarvis.face.classification import training, classifier


def main():
    parser = argparse.ArgumentParser(description='Train or use the classifier')
    parser.add_argument('mode', type=str, help='The classification mode [train/predict]', default='predict',
                        choices=['train', 'predict'])

    args = parser.parse_args()

    if args.mode == 'train':
        training.train()
    elif args.mode == 'predict':
        classifier.predict()


if __name__ == '__main__':
    main()
