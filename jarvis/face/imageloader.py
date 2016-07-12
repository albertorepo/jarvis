import collections
from os import listdir
from os.path import isdir, join

import cv2
import numpy as np


class ImageLoader:
    def __init__(self):
        pass

    def load_image_from_path(self, data_folder_path, min_images_per_folder=0):
        person_names, file_paths = [], []
        for person_name in sorted(listdir(data_folder_path)):
            folder_path = join(data_folder_path, person_name)
            if not isdir(folder_path):
                continue
            paths = [join(folder_path, f) for f in listdir(folder_path)]
            n_pictures = len(paths)
            if n_pictures >= min_images_per_folder:
                person_name = person_name.replace('_', ' ')
                person_names.extend([person_name] * n_pictures)
                file_paths.extend(paths)

        n_faces = len(file_paths)
        if n_faces == 0:
            raise ValueError("min_faces_per_person=%d is too restrictive" %
                             min_images_per_folder)

        target_names = np.unique(person_names)
        target = np.searchsorted(target_names, person_names)

        images = self._load_imgs(file_paths)
        indices = np.arange(n_faces)
        np.random.RandomState(42).shuffle(indices)
        faces, target = images[indices], target[indices]
        return Bunch(data=images, target=target, target_names=target_names)

    def _load_imgs(self, file_paths):
        images = []
        n_faces = len(file_paths)
        for i, file_path in enumerate(file_paths):
            if i % 1000 == 0:
                print "Loading face #%05d / %05d" % (i + 1, n_faces)
            img = cv2.imread(file_path)
            images.append(img)
        return np.array(images)


class Bunch(collections.namedtuple('Bunch', ['data', 'target', 'target_names'])):
    __slots__ = ()

    def __str__(self):
        return "Bunch of {} images ".format(len(self.data))
