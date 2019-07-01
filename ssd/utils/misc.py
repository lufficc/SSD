import errno
import os


def str2bool(s):
    return s.lower() in ('true', '1')


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
