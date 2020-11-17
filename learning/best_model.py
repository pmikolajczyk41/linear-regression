from functools import partial
from pathlib import Path
from random import shuffle
from typing import Tuple

from algebra import Matrix, Vector
from cost.basic_cost import squared_error
from cost.regularization import ridge, parametrize, lasso
from data.X import X
from data.basis_functions import id_basis_functions
from data.loading import load
from data.normalization import ScalingType
from learning.linear_learner import LinearLearner
from learning.parameters import Parameters
from learning.stop import StopCondition

parameters = [Parameters(squared_error, parametrize(ridge, 0.8), 0.01, None, id_basis_functions(5)),
              Parameters(squared_error, parametrize(lasso, 0.8), 0.01, None, id_basis_functions(5))]


def split2(x: Matrix, y: Vector, p: float) -> Tuple[Tuple[Matrix, Vector], Tuple[Matrix, Vector]]:
    assert len(x) == len(y) and 0. <= p <= 1.
    domain = list(zip(x, y))
    shuffle(domain)

    prefix_size = int(p * len(domain))
    prefix, suffix = domain[:prefix_size], domain[prefix_size:]
    return tuple(zip(*prefix)), tuple(zip(*suffix))


def split3(x: Matrix, y: Vector, p1: float, p2: float) -> \
        Tuple[Tuple[Matrix, Vector], Tuple[Matrix, Vector], Tuple[Matrix, Vector]]:
    (x1, y1), (x2, y2) = split2(x, y, p1)
    (x3, y3), (x4, y4) = split2(x2, y2, p2 / (1 - p1))
    return (x1, y1), (x3, y3), (x4, y4)


def find_best_model(x_train: X, y_train: Vector, x_val: X, y_val: Vector) -> Tuple[Vector, float, Parameters]:
    stop_condition = StopCondition(None, 0.000001, 2000)

    train = partial(LinearLearner().train, x_train, y_train, stop_condition)
    trained = map(lambda param: train(param), parameters)

    def evaluate(paired):
        theta, params = paired
        error = params.cost()(x_val.append_ones(), theta, y_val)[0] / x_val.nsamples()
        return error, theta, params

    evaluated = map(lambda paired: evaluate(paired), zip(trained, parameters))

    return max(evaluated, key=lambda e: e[0])


if __name__ == '__main__':
    x, y = load(Path('../noise.data'))
    x = X(x).normalize(ScalingType.MIN_MAX_1)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split3(x.by_sample(), y, 1 / 3, 1 / 3)
    print(find_best_model(X(x_train), y_train, X(x_val), y_val))
