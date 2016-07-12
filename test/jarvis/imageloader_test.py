import os

from nose.tools import assert_equal

from jarvis.face.imageloader import ImageLoader


class TestImageLoader():
    def test_list_directory(self):
        bunch = ImageLoader().load_image_from_path('/home/alberto/scikit_learn_data/lfw_home/lfw_funneled', 70)
        assert_equal(len(bunch.data), 1288)
