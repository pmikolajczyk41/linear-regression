from typing import Tuple

from cost import _mult_mv, _signum
from data import Vector, vector, Scalar
from data.X import X


def squared_error(x: X, theta: Vector, y: Vector) -> Tuple[Scalar, Vector]:
    pred = _mult_mv(x.by_sample(), theta)
    diff = vector(map(lambda p, q: p - q, zip(pred, y)))
    error = sum(map(lambda e: 0.5 * e ** 2, diff))
    grad = _mult_mv(x.by_feature(), diff)
    return error, grad


def abs_error(x: X, theta: Vector, y: Vector) -> Tuple[Scalar, Vector]:
    pred = _mult_mv(x.by_sample(), theta)
    diff = vector(map(lambda p, q: p - q, zip(pred, y)))
    error = sum(map(lambda e: abs(e), diff))

    sign = vector(map(lambda d: _signum(d), diff))
    grad = _mult_mv(x.by_feature(), sign)

    return error, grad
