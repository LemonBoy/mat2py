# type: ignore

from types import SimpleNamespace
from copy import deepcopy
from itertools import zip_longest

from mat2py.common.backends import numpy as np

from .array import M, MatArray, mp_convert_round, mp_convert_scalar, mp_try_match_shape, \
    MatIndex
from .cell import C, CellArray


class Struct(SimpleNamespace):
    def __getitem__(self, item):
        return getattr(self, item)


class StructArray:
    def __init__(self, **kwargs):
        self._base = Struct(**kwargs)
        self._items = np.ndarray((0, ), dtype=object)

    def __call__(self, item, *rest_item):
        if rest_item:
            item = (item, *rest_item)
        return self.__getitem__(item)

    def _expand(self, index):
        increase_dims = len(index) > self._items.ndim
        if increase_dims:
            # Add bogus dimensions before calling pad().
            shape = self._items.shape
            self._items = self._items.reshape(
                    tuple(shape[ii] if ii < len(shape) else 0
                          for ii in range(len(index))))
        shape_diff = [x - y for (x, y) in zip(index, self._items.shape)]
        if any(x > 0 for x in shape_diff):
            # Initialize the newly added slots with empty structures.
            def pad_fn(vector, iaxis_pad_width, iaxis, kwargs):
                for ii in range(iaxis_pad_width[0]):
                    vector[ii] = deepcopy(self._base)
                for ii in range(-iaxis_pad_width[1], len(vector)):
                    vector[ii] = deepcopy(self._base)
            self._items = np.pad(self._items,
                                 tuple((0, x) for x in shape_diff),
                                 pad_fn)

    def __getitem__(self, item):
        try:
            index = MatIndex(item)(self._items.shape)
        except ValueError:
            self._expand(item if isinstance(item, tuple) else (item,))
            index = MatIndex(item)(self._items.shape)
        return self._items[index]

    def __setitem__(self, item, value):
        try:
            index = MatIndex(item)(self._items.shape)
        except ValueError:
            self._expand(item if isinstance(item, tuple) else (item,))
            index = MatIndex(item)(self._items.shape)
        self._items[index] = value

    def __array_ufunc__(self, *args, **kwargs):
        raise NotImplementedError("cell do not support calculation")

    def __repr__(self):
        return self._items.__repr__()

    @property
    def size(self):
        return self._items.size

    @property
    def shape(self):
        return self._items.shape


def mp_fieldnames_list(a):
    assert isinstance(a, StructArray)
    return [i for i in a.__dict__.keys() if not i.startswith("__")]


def fieldnames(a, *args):
    if args:
        raise NotImplementedError("fieldnames")
    return C.__class_getitem__(*mp_fieldnames_list(a))


def struct(*args):
    if len(args) == 0:
        return StructArray()
    if len(args) == 1:
        raise NotImplementedError
    assert len(args) % 2 == 0
    names, values = args[::2], args[1::2]
    assert all(isinstance(i, str) for i in names)

    # if any(isinstance(i, CellArray) for i in values):
    #     raise NotImplementedError

    return StructArray(**dict(zip(names, values)))
