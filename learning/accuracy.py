from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean
from typing import NoReturn, Tuple, Iterable

import matplotlib.pyplot as plt

from algebra import Vector
from cost.basic_cost import squared_error, abs_error
from cost.regularization import ridge, lasso
from data.X import X
from data.basis_functions import id_basis_functions, second_degree_basis_functions
from data.loading import load
from data.normalization import ScalingType
from data.splitting import split2
from learning.model import Model
from learning.parameters import Parameters

# number of trials before deciding about optimality/accuracy
REPETITIONS = 5

models = [
    Model(squared_error, ridge, id_basis_functions(5)),
    Model(squared_error, lasso, id_basis_functions(5)),
    Model(squared_error, ridge, second_degree_basis_functions(5)),
    Model(squared_error, lasso, second_degree_basis_functions(5)),
    Model(abs_error, ridge, id_basis_functions(5)),
    Model(abs_error, lasso, second_degree_basis_functions(5)),
]

parameters = [
    Parameters(tuple([reg_par]), step, stdev)
    for reg_par, step, stdev in product([0.01, 0.1, 0.5, 2.], [0.05, 0.1, 0.7], [0., 1., 2.])
]


def report_best_parameters(model: Model, theta: Vector, train_error: float, test_error: float) -> NoReturn:
    repr = str(model).replace(', ', '\n\t').strip('{}')
    print(f'Reached best parameters for a model: \n\t{repr}\n'
          f'    producing a hypothesis: {theta}\n'
          f'    with an average error on train set: {train_error}\n'
          f'    with an error on test set: {test_error}')


def optimize_model(model: Model, train_set, test_set) -> Model:
    (x, y), (x_test, y_test) = train_set, test_set
    x_test = X(x_test)

    accuracies = defaultdict(list)
    for _ in range(REPETITIONS):
        (x_train, y_train), (x_val, y_val) = split2(x, y, 1 / 2)
        x_train, x_val = X(x_train), X(x_val)

        evaluated = model.evaluate_parameters(x_train, y_train, x_val, y_val, parameters)
        for e, _, p in evaluated:
            accuracies[p].append(e)

    accuracies = [(p, mean(es)) for p, es in accuracies.items()]
    best_params, train_error = min(accuracies, key=lambda x: x[1])

    model.set_parameters(best_params)
    theta = model.train(X(x), y)
    test_error = model.error(x_test, y_test)

    report_best_parameters(model, theta, train_error, test_error)
    return model


def plot_learning_curve(model: Model, data: Tuple[Iterable, Iterable]) -> NoReturn:
    plt.ylim(0, 30)
    plt.plot(*data)
    repr = str(model).replace(', ', '\n').strip('{}').replace('\'', '')
    plt.title(repr)
    plt.show()


if __name__ == '__main__':
    x, y = load(Path('../noise.data'))
    x = X(x).normalize(ScalingType.MIN_MAX_1)
    train, test = split2(x.by_sample(), y, 1 / 3)

    optimized_models = (optimize_model(m, train, test)
                        for m in models)

    train_fractions = [0.01, 0.02, 0.03, 0.125, 0.625, 1.]

    for model in optimized_models:
        accuracies = defaultdict(list)
        for tf, _ in product(train_fractions, range(REPETITIONS)):
            (x_train, y_train), (x_test, y_test) = split2(x.by_sample(), y, tf)
            if tf == 1.:
                x_test, y_test = x_train, y_train
            x_train, x_test = X(x_train), X(x_test)

            model.train(x_train, y_train)
            accuracies[tf].append(model.error(x_test, y_test))

        means = [mean(accuracies[tf]) for tf in train_fractions]
        for tf, mn in zip(train_fractions, means):
            print(f'Accuracy on {tf * 100}% samples: {mn:.4f}')

        plot_learning_curve(model, (train_fractions, means))
