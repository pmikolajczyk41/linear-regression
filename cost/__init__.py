from functools import reduce

from data import Vector, vector, Matrix, Scalar


def _mult_vv(u: Vector, v: Vector) -> Scalar:
    assert len(u) == len(v)
    return reduce(lambda acc, a, b: acc + a * b, zip(u, v), 0)


def _mult_mv(m: Matrix, v: Vector) -> Vector:
    assert len(m[0]) == len(v)
    return vector(map(lambda u: _mult_vv(u, v), m))


def _signum(x: float) -> float:
    if x < 0.: return -1.
    if x > 0.: return 1.
    return x
