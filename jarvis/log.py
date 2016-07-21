import logging
import logging.handlers
import os
from ConfigParser import ConfigParser

import sys


def init_logger(tag):
    logger = logging.getLogger(tag)

    file_dir = os.path.dirname(os.path.realpath(__file__))

    config = ConfigParser()
    try:
        config.read(os.path.join(file_dir, '../log_config.ini'))
    except:
        raise Exception("Can't find face_config-ini file.")

    logger.setLevel(config.get("Logging", "Level"))

    fh = logging.handlers.RotatingFileHandler(
        config.get("Logging", "Path"), mode='w', maxBytes=int(config.get("Logging", "FileMaxBytes")),
        backupCount=int(config.get("Logging", "FileCount"))
    )
    fh.setLevel(config.get("Logging", "FileLevel"))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(config.get("Logging", "StreamLevel"))

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logging.getLogger(tag)