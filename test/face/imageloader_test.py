import os

from nose.tools import assert_equal, assert_true, assert_raises

from jarvis.face.imageloader import ImageLoader


class TestImageLoader():
    def setUp(self):
        self.image_loader = ImageLoader('/Users/albertocastano/development/lfw_funneled')

    def test_load_image(self):
        self.image_loader.load_images()
        assert_true(self.image_loader.data)
        assert_equal(len(self.image_loader.data.images), len(self.image_loader.data.target))

    def test_images_are_loaded(self):
        self.image_loader.load_images_from_path('/Users/albertocastano/development/lfw_funneled')
        assert_true(self.image_loader.data)
        assert_equal(len(self.image_loader.data.images), len(self.image_loader.data.target))

    def test_no_image_is_loaded(self):
        assert_raises(ValueError,
                      self.image_loader.load_images_from_path, '/Users/albertocastano/development/lfw_funneled',
                      min_images_per_folder=99999999999)

    def test_minimum_cannot_be_grater_than_maximum(self):
        assert_raises(ValueError, self.image_loader.load_images_from_path,
                      '/Users/albertocastano/development/lfw_funneled',
                      min_images_per_folder=10, max_images_per_folder=9)

    def test_maximum_number_of_images_too_big(self):
        self.image_loader.load_images_from_path('/Users/albertocastano/development/lfw_funneled',
                                               max_images_per_folder=9999999999999)
        assert_true(self.image_loader.data)
        assert_equal(len(self.image_loader.data.images), len(self.image_loader.data.target))

    def test_preprocessing_without_images(self):
        assert_raises(AttributeError, self.image_loader.preprocessing)

    def test_preprocessing(self):
        self.image_loader.load_images_from_path('/Users/albertocastano/development/lfw_funneled',
                      min_images_per_folder=10, max_images_per_folder=10)
        self.image_loader.preprocessing('cascade.xml')
        images_preprocessed = self.image_loader.images
        assert_true(images_preprocessed)
        assert_equal(len(images_preprocessed.images), len(images_preprocessed.target))
