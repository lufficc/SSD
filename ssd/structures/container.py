class Container:
    def __init__(self, *args, **kwargs):
        self._data_dict = dict(*args, **kwargs)

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
        """resize boxes
        Args:
            size: (width, height)
        Returns:
            self
        """
        img_width = getattr(self, 'img_width', -1)
        img_height = getattr(self, 'img_height', -1)
        assert img_width > 0 and img_height > 0
        assert 'boxes' in self._data_dict
        boxes = self._data_dict['boxes']
        new_width, new_height = size
        boxes[:, 0::2] *= (new_width / img_width)
        boxes[:, 1::2] *= (new_height / img_height)
        return self

    def __repr__(self):
        return self._data_dict.__repr__()
