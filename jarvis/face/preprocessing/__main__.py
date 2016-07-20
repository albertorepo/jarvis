import os
import random

import cv2

from jarvis.face.preprocessing import utils
from jarvis.face.preprocessing.aligndlib import AlignDlib
from jarvis.face.preprocessing.imageloader import ImageLoader
from jarvis.face.utils import get_config


def main():
    raw_images_path = get_config().get('PreProcessing', 'RawImagesPath')
    images = load_images(raw_images_path)
    align_images_and_save_them(images, os.path.join(raw_images_path, '..', 'aligned'))


def load_images(path):
    images = list(ImageLoader(path).load_images())
    random.shuffle(images)
    return images


def align_images_and_save_them(images, output_dir):
    landmark_indices = AlignDlib.OUTER_EYES_AND_NOSE
    aligner = AlignDlib(get_config().get('PreProcessing', 'FacePredictorPath'))

    n_fallbacks = 0
    n_images = len(images)

    for number, image in enumerate(images):
        if number % 1000 == 0:
            print "Aligned {}/{} images".format(number, n_images)
        out_dir = os.path.join(output_dir, image.label)
        utils.mkdir(out_dir)
        outputPrefix = os.path.join(out_dir, image.name)
        imgName = outputPrefix + ".png"
        rgb = image.to_rgb()
        img_dim = get_config().get('PreProcessing', 'FaceSize')
        out_rgb = aligner.align(int(img_dim), rgb, landmarkIndices=landmark_indices)
        if out_rgb is None:
            # print image.path
            # print "  + Unable to align."
            n_fallbacks += 1
            # TODO: Implement fallback
            out_rgb = rgb

        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(imgName, out_bgr)

    print 'Number of fallbacks: ', n_fallbacks


if __name__ == '__main__':
    main()
