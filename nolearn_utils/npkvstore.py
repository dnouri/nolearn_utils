import os
import cPickle as pickle
import numpy as np


def add_suffix(path, suffix):
    """Add suffix to the file name of the path"""
    parts = path.split(os.path.sep)
    dirname = os.path.join(*parts[:-1])
    fname_ext = parts[-1]
    fname, ext = os.path.splitext(fname_ext)

    fname = fname + suffix

    fname_ext = fname + ext
    path = os.path.join(dirname, fname_ext)
    return path


class NPKVStore(object):
    """Key value store backed by numpy memmap"""

    def __init__(self, mmap_fname, dtype, mode, shape):
        self.mmap_fname = mmap_fname
        self.key_fname = add_suffix(mmap_fname, '_idx')
        self.dtype = dtype
        self.mode = mode
        self.shape = shape

        self.mmap = np.memmap(
            self.mmap_fname,
            dtype=self.dtype, mode=self.mode, shape=self.shape
        )

        self.key_map = {}
        self._index = 0

    def _key_to_index(self, key):
        return self.key_map[key]

    def _keys_to_indices(self, keys):
        return [self._key_to_index(key) for key in keys]

    def _add_key(self, key):
        self.key_map[key] = self._index
        self._index += 1

    def _remove_key(self, key):
        del self.key_map[key]

    def __getitem__(self, keys):
        if type(keys) == list:
            return self.mmap[self._keys_to_indices(keys)]
        else:
            return self.mmap[self._key_to_index(keys)]

    def __setitem__(self, keys, values):
        if type(keys) == list:
            for key in keys:
                self._add_key(key)
            self.mmap[self._keys_to_indices(keys)] = values
        else:
            self._add_key(keys)
            self.mmap[self._key_to_index(keys)] = values

    def __contains__(self, key):
        return key in self.key_map

    def remove(self, key):
        self.mmap[self._key_to_index(key)] = 0
        self._remove_key(key)

    def flush(self):
        self.mmap.flush()
        pickle.dump(
            self.key_map, open(self.key_fname, self.mode),
            pickle.HIGHEST_PROTOCOL
        )
