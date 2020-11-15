from functools import reduce
from typing import Callable, Tuple

from data import Vector, Scalar, Matrix, vector
from data.X import X

BasicCost = Cost = Callable[[X, Vector, Vector], Tuple[Scalar, Vector]]
Regularization = Callable[[Vector], Tuple[Scalar, Vector]]


def _mult_vv(u: Vector, v: Vector) -> Scalar:
    assert len(u) == len(v)
    return reduce(lambda acc, a, b: acc + a * b, zip(u, v), 0)


def _mult_mv(m: Matrix, v: Vector) -> Vector:
    assert len(m[0]) == len(v)
    return vector(map(lambda u: _mult_vv(u, v), m))


def _sum_vv(u: Vector, v: Vector) -> Vector:
    assert len(u) == len(v)
    return vector(map(lambda a, b: a + b, zip(u, v)))


def _diff_vv(u: Vector, v: Vector) -> Vector:
    assert len(u) == len(v)
    return vector(map(lambda a, b: a - b, zip(u, v)))


def _signum(x: float) -> float:
    if x < 0.: return -1.
    if x > 0.: return 1.
    return x
