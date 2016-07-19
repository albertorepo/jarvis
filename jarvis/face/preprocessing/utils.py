import os

import errno


def mkdir(dir):
    assert dir is not None

    try:
        os.makedirs(dir)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(dir):
            pass
        else:
            raise
