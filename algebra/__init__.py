from functools import reduce
from operator import sub, mul
from typing import Tuple

Scalar = float
Vector = Tuple[Scalar, ...]
Matrix = Tuple[Vector, ...]

vector = tuple


def mult_vs(u: Vector, s: Scalar) -> Vector:
    return vector(map(lambda x: s * x, u))


def mult_v(u: Vector) -> Scalar:
    return reduce(lambda acc, x: acc + x * x, u, 0)


def mult_vv(u: Vector, v: Vector) -> Scalar:
    assert len(u) == len(v)
    return reduce(lambda acc, uv: acc + mul(*uv), zip(u, v), 0)


def mult_mv(m: Matrix, v: Vector) -> Vector:
    assert len(m[0]) == len(v)
    return vector(map(lambda u: mult_vv(u, v), m))


def sum_vv(u: Vector, v: Vector) -> Vector:
    assert len(u) == len(v)
    return vector(map(sum, zip(u, v)))


def diff_vv(u: Vector, v: Vector) -> Vector:
    assert len(u) == len(v)
    return vector(map(lambda x: sub(*x), zip(u, v)))


def signum(x: float) -> float:
    if x < 0.: return -1.
    if x > 0.: return 1.
    return x
