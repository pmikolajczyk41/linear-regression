from itertools import combinations_with_replacement as pairs
from typing import Callable, Iterable

from algebra import Vector, Scalar

BasisFunction = Callable[[Vector], Scalar]


class BasisFunctions(tuple):
    def __new__(cls, description: str, functions: Iterable[BasisFunction]):
        cls.__description__ = description
        return tuple.__new__(BasisFunctions, functions)


def id_basis_functions(nfeatures: int) -> BasisFunctions:
    return BasisFunctions('Identity', map(lambda i: lambda s: s[i], range(nfeatures)))


def second_degree_basis_functions(nfeatures: int) -> BasisFunctions:
    linear = id_basis_functions(nfeatures)
    quadratic = tuple(map(lambda p: lambda s: s[p[0]] * s[p[1]], pairs(range(nfeatures), 2)))
    return BasisFunctions('Second degree', linear + quadratic)
