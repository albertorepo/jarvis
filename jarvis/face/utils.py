from ConfigParser import ConfigParser


def get_config():
    config = ConfigParser()
    try:
        config.read('../../face_config.ini')
    except:
        raise Exception("Can't find face_config-ini file.")

    return config
