import collections
from os import listdir
from os.path import isdir, join

import cv2
import numpy as np


class ImageLoader:
    def __init__(self, images_path=None):
        self.images_path = images_path
        self.data = None

    def load_images_from_path(self, data_folder_path, min_images_per_folder=0, max_images_per_folder=np.inf):
        person_names, file_paths = [], []
        for person_name in sorted(listdir(data_folder_path)):
            folder_path = join(data_folder_path, person_name)
            if not isdir(folder_path):
                continue
            paths = [join(folder_path, f) for f in listdir(folder_path)]
            n_pictures = len(paths)
            if n_pictures >= min_images_per_folder:
                person_name = person_name.replace('_', ' ')
                if person_name == "Alberto Castano":
                    person_names.extend([person_name] * n_pictures)
                    file_paths.extend(paths)
                else:
                    limit = int(np.min([n_pictures, max_images_per_folder]))
                    person_names.extend([person_name] * limit)
                    file_paths.extend(paths[0:limit])

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
        self.data = Bunch(images=faces, target=target, target_names=target_names)

    def load_images(self, min_images_per_folder=0, max_images_per_folder=np.inf):
       self.load_images_from_path(self.images_path, min_images_per_folder, max_images_per_folder)

    def _load_imgs(self, file_paths):
        images = []
        n_faces = len(file_paths)
        for i, file_path in enumerate(file_paths):
            if i % 1000 == 0:
                print "Loading face #%05d / %05d" % (i + 1, n_faces)
            img = cv2.imread(file_path)
            images.append(img)
        return np.asarray(images)

    def preprocessing(self, casc_path):

        if not self.data:
            raise AttributeError("Load some images first")
        faces_data = []
        target = self.data.target

        face_cascade = cv2.CascadeClassifier(casc_path)
        for (ind, img) in enumerate(self.data.images):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                img,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            try:
                (x, y, w, h) = faces[0]
                area = w * h
                for (xt, yt, wt, ht) in faces:
                    if wt * ht > area:
                        (x, y, w, h) = (xt, yt, wt, ht)

            except IndexError:
                print "WARNING - Face not detected"
                target[ind] = -1
                continue

            face = img[y:y + h, x:x + w]
            faces_data.append(face)

        self.data = Bunch(images=faces_data,
                            target=self.data.target[np.where(self.data.target != -1)],
                            target_names=self.data.target_names)


class Bunch(collections.namedtuple('Bunch', ['images', 'target', 'target_names'])):
    __slots__ = ()

    def __str__(self):
        return "Bunch of {} images ".format(len(self.images))
