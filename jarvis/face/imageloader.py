import os

from jarvis.face.image import Image


class ImageLoader:
    def __init__(self, path):
        self.path = path

    def load_images(self):
        assert self.path is not None

        valid_extensions = ['.jpg', '.png']

        for subdir, dirs, files in os.walk(self.path):
            for path in files:
                image_class, file_name = (os.path.basename(subdir), path)
                image_name, extension = os.path.splitext(file_name)
                if extension in valid_extensions:
                    yield Image(image_class, image_name, os.path.join(subdir, file_name))
