import os
from ConfigParser import ConfigParser

import errno


def get_config():
    file_dir = os.path.dirname(os.path.realpath(__file__))

    config = ConfigParser()
    try:
        config.read(os.path.join(file_dir, '../../face_config.ini'))
    except:
        raise Exception("Can't find face_config-ini file.")

    return config

def mkdir(dir):
    assert dir is not None

    try:
        os.makedirs(dir)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(dir):
            pass
        else:
            raise

