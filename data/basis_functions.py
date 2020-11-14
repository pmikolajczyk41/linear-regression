from itertools import combinations_with_replacement as pairs
from typing import Callable, Tuple

from data import Vector, Scalar

BasisFunction = Callable[[Vector], Scalar]


def id_basis_functions(nfeatures: int) -> Tuple[BasisFunction, ...]:
    return tuple(map(lambda i: lambda s: s[i], range(nfeatures)))


def second_degree_basis_functions(nfeatures: int) -> Tuple[BasisFunction, ...]:
    linear = id_basis_functions(nfeatures)
    quadratic = tuple(map(lambda p: lambda s: s[p[0]] * s[p[1]], pairs(range(nfeatures), 2)))
    return linear + quadratic
