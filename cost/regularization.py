from functools import reduce
from typing import Tuple

from cost import _mult_vv, _signum
from data import Vector, Scalar, vector


def lasso(theta: Vector, lmb: Scalar) -> Tuple[Scalar, Vector]:
    cost = lmb * reduce(lambda acc, t: acc + abs(t), theta, 0)
    grad = lmb * reduce(lambda acc, t: acc + _signum(t), 0)
    return cost, grad


def ridge(theta: Vector, lmb: Scalar) -> Tuple[Scalar, Vector]:
    cost = lmb * _mult_vv(theta, theta)
    grad = vector(map(lambda t: 2 * lmb * t, theta))
    return cost, grad


def elastic_net(theta: Vector, lmb1: Scalar, lmb2: Scalar) -> Tuple[Scalar, Vector]:
    cl, gl = lasso(theta, lmb1)
    cr, gr = ridge(theta, lmb2)
    grad = vector(map(lambda l, r: l + r, zip(gl, gr)))
    return cl + cr, grad
