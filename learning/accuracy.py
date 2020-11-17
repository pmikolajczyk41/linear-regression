from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean
from typing import NoReturn

from algebra import Vector
from cost.basic_cost import squared_error, abs_error
from cost.regularization import ridge, lasso
from data.X import X
from data.basis_functions import id_basis_functions, second_degree_basis_functions
from data.loading import load
from data.normalization import ScalingType
from data.splitting import split3, split2
from learning.model import Model
from learning.parameters import Parameters

models = [
    Model(squared_error, ridge, id_basis_functions(5)),
    Model(squared_error, lasso, id_basis_functions(5)),
    Model(squared_error, ridge, second_degree_basis_functions(5)),
    Model(squared_error, lasso, second_degree_basis_functions(5)),
    Model(abs_error, ridge, id_basis_functions(5)),
    Model(abs_error, lasso, second_degree_basis_functions(5)),
]

parameters = [
    Parameters(tuple([reg_par]), step, init_theta)
    for reg_par, step, init_theta in product([0.0, 0.4, 0.8, 1.2], [0.1, 0.05, 0.01, 0.005, 0.001], [None])
]


def report_best_parameters(model: Model, theta: Vector, error: float) -> NoReturn:
    repr = str(model).replace(', ', '\n\t').strip('{}')
    print(f'Reached best parameters for a model: \n\t{repr}\n'
          f'    producing a hypothesis: {theta}\n'
          f'    with an average error: {error}')


def optimize_model(model: Model, train_set, val_set, test_set) -> Model:
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = train_set, val_set, test_set
    x_train, x_val, x_test = X(x_train), X(x_val), X(x_test)
    model.set_best_parameters(x_train, y_train, x_val, y_val, parameters)

    theta = model.train(x_train, y_train)
    error = model.error(x_test, y_test)
    report_best_parameters(model, theta, error)

    return model


if __name__ == '__main__':
    x, y = load(Path('../noise.data'))
    x = X(x).normalize(ScalingType.MIN_MAX_1)
    train, val, test = split3(x.by_sample(), y, 1 / 3, 1 / 3)

    optimized_models = (optimize_model(m, train, val, test)
                        for m in (models[:1]))

    for model in optimized_models:
        accuracies = defaultdict(list)
        for train_fraction, _ in product([0.01, 0.02, 0.03, 0.125, 0.625, 1.], range(5)):
            (x_train, y_train), (x_test, y_test) = split2(x.by_sample(), y, train_fraction)
            if train_fraction == 1.:
                x_test, y_test = x_train, y_train
            x_train, x_test = X(x_train), X(x_test)

            model.train(x_train, y_train)
            accuracies[train_fraction].append(model.error(x_test, y_test))

        for train_fraction in [0.01, 0.02, 0.03, 0.125, 0.625, 1.]:
            print(f'Accuracy on {train_fraction * 100}% samples: {mean(accuracies[train_fraction]):.4f}')
