import numpy as np
import os

DIRNAME = '~/.nolearn_utils/datasets'
DIRNAME = os.path.expanduser(DIRNAME)


class BaseDataset(object):
    def __init__(self, dirname=''):
        pass

    def cache(self, image_size):
        pass


class CIFAR10(BaseDataset):
    name = 'cifar10'

    @property
    def X(self):
        pass

    @property
    def y(self):
        pass
