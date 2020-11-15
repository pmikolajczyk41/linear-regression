from functools import reduce
from typing import Tuple, Callable, Any

from algebra import Vector, Scalar, vector, sum_vv, signum, mult_v
from cost import Regularization


def lasso(theta: Vector, lmb: Scalar) -> Tuple[Scalar, Vector]:
    # ignore bias
    cost = lmb * reduce(lambda acc, t: acc + abs(t), theta[:-1], 0)
    grad = vector(map(lambda t: lmb * signum(t), theta[:-1])) + (0.,)
    return cost, grad


def ridge(theta: Vector, lmb: Scalar) -> Tuple[Scalar, Vector]:
    # ignore bias
    cost = lmb * mult_v(theta[:-1])
    grad = vector(map(lambda t: 2 * lmb * t, theta[:-1])) + (0.,)
    return cost, grad


def elastic_net(theta: Vector, lmb1: Scalar, lmb2: Scalar) -> Tuple[Scalar, Vector]:
    cl, gl = lasso(theta, lmb1)
    cr, gr = ridge(theta, lmb2)
    return cl + cr, sum_vv(gl, gr)


def parametrize(reg: Callable[[Vector, Any], Tuple[Scalar, Vector]], *parameters, **named_parameters) -> Regularization:
    def parametrized(theta: Vector) -> Tuple[Scalar, Vector]:
        return reg(theta, *parameters, **named_parameters)

    return parametrized
