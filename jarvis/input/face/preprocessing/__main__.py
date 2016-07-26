import os
import random

from jarvis.face.preprocessing.imageloader import ImageLoader
from jarvis.face.preprocessing.preprocesser import PreProcessor

from jarvis.input.face.utils import get_config


def main():
    raw_images_path = get_config().get('PreProcessing', 'RawImagesPath')
    preprocessor = PreProcessor()
    images = load_images(raw_images_path)
    preprocessor.align_images_and_save_them(images, os.path.join(raw_images_path, '..', 'aligned'))


def load_images(path):
    images = list(ImageLoader(path).load_images())
    random.shuffle(images)
    return images


if __name__ == '__main__':
    main()
