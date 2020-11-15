from itertools import repeat
from typing import Tuple

from algebra import Matrix, vector
from data.basis_functions import BasisFunction
from data.normalization import ScalingType, normalize


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
        return vector(zip(*self._data))

    def normalize(self, scaling_type: ScalingType) -> 'X':
        normalized_vectors = map(lambda f: normalize(f, scaling_type), self.by_feature())
        return X(vector(zip(*normalized_vectors)))

    def convert(self, basis_functions: Tuple[BasisFunction, ...]) -> 'X':
        sample_mapper = lambda s: vector(map(lambda bf: bf(s), basis_functions))
        samples = map(lambda sample: sample_mapper(sample), self._data)
        return X(vector(samples))

    def append_ones(self) -> 'X':
        ones = vector(repeat(1., self._m))
        extended = self.by_feature() + (ones,)
        return X(vector(zip(*extended)))
