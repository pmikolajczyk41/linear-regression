from pathlib import Path

from cost.basic_cost import squared_error
from cost.regularization import ridge
from data.X import X
from data.basis_functions import id_basis_functions
from data.loading import load
from data.normalization import ScalingType
from data.splitting import split3
from learning.model import Model
from learning.parameters import Parameters

if __name__ == '__main__':
    x, y = load(Path('../noise.data'))
    x = X(x).normalize(ScalingType.MIN_MAX_1)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split3(x.by_sample(), y, 1 / 3, 1 / 3)

    m1 = Model(squared_error, ridge, id_basis_functions(5))
    parameters = [Parameters((0.8,), 0.1)]
    m1.set_best_parameters(X(x_train), y_train, X(x_val), y_val, parameters)
