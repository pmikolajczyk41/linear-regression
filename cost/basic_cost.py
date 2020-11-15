from typing import Tuple

from algebra import Vector, vector, Scalar, mult_mv, diff_vv, signum
from data.X import X


def squared_error(x: X, theta: Vector, y: Vector) -> Tuple[Scalar, Vector]:
    pred = mult_mv(x.by_sample(), theta)
    diff = diff_vv(pred, y)
    error = sum(map(lambda e: 0.5 * e ** 2, diff))
    grad = mult_mv(x.by_feature(), diff)
    return error, grad


def abs_error(x: X, theta: Vector, y: Vector) -> Tuple[Scalar, Vector]:
    pred = mult_mv(x.by_sample(), theta)
    diff = diff_vv(pred, y)
    error = sum(map(lambda e: abs(e), diff))

    sign = vector(map(lambda d: signum(d), diff))
    grad = mult_mv(x.by_feature(), sign)

    return error, grad
