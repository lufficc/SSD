from typing import Mapping

from matplotlib import collections


class Container:
    def __init__(self, **kwargs):
        self._data_dict = {**kwargs}

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)

    def __getitem__(self, key):
        return self._data_dict[key]

    def __iter__(self):
        return self._data_dict.__iter__()

    def __setitem__(self, key, value):
        self._data_dict[key] = value

    def to(self, *args, **kwargs):
        keys = list(self._data_dict.keys())
        for key in keys:
            value = self._data_dict[key]
            if hasattr(value, 'to'):
                self._data_dict[key] = value.to(*args, **kwargs)
        return self

    def numpy(self):
        keys = list(self._data_dict.keys())
        for key in keys:
            value = self._data_dict[key]
            if hasattr(value, 'numpy'):
                self._data_dict[key] = value.numpy()
        return self

    def resize(self, size):
        img_width = getattr(self, 'img_width', -1)
        img_height = getattr(self, 'img_height', -1)
        if img_width > 0 and img_height > 0:
            new_width, new_height = size
            if 'boxes' in self._data_dict:
                self._data_dict['boxes'][:, 0::2] = self._data_dict['boxes'][:, 0::2] / img_width * new_width
                self._data_dict['boxes'][:, 1::2] = self._data_dict['boxes'][:, 1::2] / img_height * new_height
        return self

    def __repr__(self):
        return self._data_dict.__repr__()
