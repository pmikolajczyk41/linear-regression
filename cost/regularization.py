from functools import reduce, partial
from typing import Tuple, Callable

from cost import _mult_vv, _signum, Regularization, _sum_vv
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
    return cl + cr, _sum_vv(gl, gr)


def parametrize(reg: Callable[[Vector, ...], Tuple[Scalar, Vector]], *parameters, **named_parameters) -> Regularization:
    def parametrized(theta: Vector) -> Tuple[Scalar, Vector]:
        return reg(theta, *parameters, **named_parameters)

    return parametrized
