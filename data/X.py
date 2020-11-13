from typing import Tuple

from data.normalization import ScalingType, normalize

Matrix = Tuple[Tuple[float]]


class X:
    def __init__(self, data: Matrix):
        k = len(data[0])
        assert all(map(lambda s: len(s) == k, data))

        self._data = data
        self._m = len(data)
        self._k = k

    def nsamples(self) -> int:
        return self._m

    def nfeatures(self) -> int:
        return self._k

    def by_sample(self) -> Matrix:
        return self._data

    def by_feature(self) -> Matrix:
        return tuple(zip(*self._data))

    def normalize(self, scaling_type: ScalingType) -> 'X':
        normalized_vectors = map(lambda f: normalize(f, scaling_type), self.by_feature())
        return X(tuple(zip(*normalized_vectors)))
