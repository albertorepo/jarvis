import os

import cv2

from jarvis.face import utils
from jarvis.face.preprocessing.aligndlib import AlignDlib
from jarvis.face.utils import get_config
from jarvis.log import init_logger


class PreProcessor:
    def __init__(self):
        self.logger = init_logger(self.__class__.__name__)

    def align_images_and_save_them(self, images, output_dir):
        landmark_indices = AlignDlib.OUTER_EYES_AND_NOSE
        aligner = AlignDlib(get_config().get('PreProcessing', 'FacePredictorPath'))

        n_fallbacks = 0
        n_images = len(images)

        for number, image in enumerate(images):
            if number % 1000 == 0:
                self.logger.info("Aligned {}/{} images".format(number, n_images))
            out_dir = os.path.join(output_dir, image.label)
            utils.mkdir(out_dir)
            outputPrefix = os.path.join(out_dir, image.name)
            imgName = outputPrefix + ".png"
            rgb = image.to_rgb()
            img_dim = get_config().get('PreProcessing', 'FaceSize')
            out_rgb = aligner.align(int(img_dim), rgb, landmarkIndices=landmark_indices)
            if out_rgb is None:
                n_fallbacks += 1
                out_rgb = rgb

            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(imgName, out_bgr)

        self.logger.warning('Number of fallbacks: ', n_fallbacks)
