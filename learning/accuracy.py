from itertools import product
from pathlib import Path

from cost.basic_cost import squared_error, abs_error
from cost.regularization import ridge, lasso
from data.X import X
from data.basis_functions import id_basis_functions, second_degree_basis_functions
from data.loading import load
from data.normalization import ScalingType
from data.splitting import split3
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


def check_model(model: Model, train_set, val_set, test_set):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = train_set, val_set, test_set
    x_train, x_val, x_test = X(x_train), X(x_val), X(x_test)
    model.set_best_parameters(x_train, y_train, x_val, y_val, parameters)

    theta = model.train(x_train, y_train)
    cum_error, _ = model.cost()(x_test.append_ones(), theta, y_test)
    print(f'Accuracy: {cum_error / x_test.nsamples():.4f}')


if __name__ == '__main__':
    x, y = load(Path('../noise.data'))
    x = X(x).normalize(ScalingType.MIN_MAX_1)
    train, val, test = split3(x.by_sample(), y, 1 / 3, 1 / 3)

    for m in models[:1]:
        check_model(m, train, val, test)
