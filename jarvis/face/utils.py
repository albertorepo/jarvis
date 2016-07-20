import os
from ConfigParser import ConfigParser


def get_config():
    file_dir = os.path.dirname(os.path.realpath(__file__))

    config = ConfigParser()
    try:
        config.read(os.path.join(file_dir, '../../face_config.ini'))
    except:
        raise Exception("Can't find face_config-ini file.")

    return config
