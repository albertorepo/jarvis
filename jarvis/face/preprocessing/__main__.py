import os
import random

import cv2
from jarvis.face.preprocessing.aligndlib import AlignDlib

from jarvis.face.preprocessing import utils
from jarvis.face.preprocessing.imageloader import ImageLoader


def main():
    # TODO: Parametrize raw_images_path
    raw_images_path = '/Users/albertocastano/development/lfw_funneled'
    images = load_images(raw_images_path)
    align_images_and_save_them(images, os.path.join(raw_images_path, '..', 'aligned'))


def load_images(path):
    images = list(ImageLoader(path).load_images())
    random.shuffle(images)
    return images


def align_images_and_save_them(images, output_dir):
    landmark_indices = AlignDlib.OUTER_EYES_AND_NOSE
    aligner = AlignDlib('./models/dlib/shape_predictor_68_face_landmarks.dat')

    n_fallbacks = 0
    n_images = len(images)

    for number, image in enumerate(images):
        if number % 1000 == 0:
            print "Aligned {}/{} images".format(number, n_images)
        out_dir = os.path.join(output_dir, image.label)
        utils.mkdir(out_dir)
        outputPrefix = os.path.join(out_dir, image.name)
        imgName = outputPrefix + ".png"
        # TODO: Parametrize the size
        rgb = image.to_rgb()
        out_rgb = aligner.align(96, rgb, landmarkIndices=landmark_indices)
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
