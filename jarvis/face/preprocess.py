import os
import random

import cv2

from jarvis.face import utils
from jarvis.face.imageloader import ImageLoader
from jarvis.face.aligndlib import AlignDlib


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

    for image in images:
        print "=== {} ===".format(image.path)
        out_dir = os.path.join(output_dir, image.label)
        utils.mkdir(out_dir)
        outputPrefix = os.path.join(out_dir, image.name)
        imgName = outputPrefix + ".png"
        # TODO: Parametrize the size
        rgb = image.to_rgb()
        out_rgb = aligner.align(96, rgb, landmarkIndices=landmark_indices)
        if out_rgb is None:
            print "  + Unable to align."
            n_fallbacks += 1
            # TODO: Implement fallback

        else:
            print "  + Aligned."
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(imgName, out_bgr)

    print 'Number of fallbacks: ', n_fallbacks


if __name__ == '__main__':
    main()
