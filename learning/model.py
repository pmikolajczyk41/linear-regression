from functools import partial
from itertools import repeat
from statistics import mean
from typing import NoReturn, Iterable

from algebra import Vector, vector, mult_vs, sum_vv
from cost import BasicCost, UnparameterizedRegularization, Cost
from cost.cost import make_cost
from cost.regularization import parametrize
from data.X import X
from data.basis_functions import BasisFunctions
from learning.parameters import Parameters
from learning.stop import StopCondition


class Model:
    stop_condition = StopCondition(None, 0.000001, 2000)

    def __init__(self, basic_cost: BasicCost,
                 regularization: UnparameterizedRegularization,
                 basis_functions: BasisFunctions):
        self._basic_cost = basic_cost
        self._regularization = regularization
        self._basis_functions = basis_functions

        self._parameters = None

    def set_best_parameters(self,
                            x_train: X, y_train: Vector, x_val: X, y_val: Vector,
                            parameters: Iterable[Parameters]):
        x_train = x_train.convert(self._basis_functions)
        x_val = x_val.convert(self._basis_functions)

        cost = lambda ps: make_cost(self._basic_cost, parametrize(self._regularization, ps))

        train = partial(self._train, x_train, y_train)
        trained = map(lambda p: train(cost(p.regularization_parameters), p.gradient_step, p.init_theta), parameters)

        def evaluate(paired):
            theta, params = paired
            cum_error, _ = cost(params.regularization_parameters)(x_val.append_ones(), theta, y_val)
            error = cum_error / x_val.nsamples()
            return error, theta, params

        evaluated = map(lambda paired: evaluate(paired), zip(trained, parameters))

        _, _, params = min(evaluated, key=lambda e: e[0])
        self._parameters = params

    def set_stop_condition(self, stop_condition: StopCondition) -> NoReturn:
        self.stop_condition = stop_condition

    def train(self, x: X, y: Vector) -> Vector:
        if self._parameters is None:
            raise RuntimeError('Parameters not set yet')

        x.convert(self._basis_functions)
        reg = parametrize(self._regularization, self._parameters.regularization_parameters)
        cost = make_cost(self._basic_cost, reg)

        return self._train(x, y, cost, self._parameters.gradient_step, self._parameters.init_theta)

    def _train(self, x: X, y: Vector, cost: Cost, step: float, init_theta: Vector = None) -> Vector:
        x = x.append_ones()
        m = x.nsamples()

        if init_theta is None:
            theta = vector(repeat(0., x.nfeatures() - 1)) + (mean(y),)
        else:
            assert len(init_theta) == x.nfeatures()
            theta = init_theta

        stop_condition = self.stop_condition
        while True:
            error, gradient = cost(x, theta, y)
            error, gradient = error / m, mult_vs(gradient, 1 / m)

            theta = sum_vv(theta, mult_vs(gradient, -step))

            stop_condition, stop = stop_condition.update(gradient, error)
            if stop:
                break
            if stop_condition._iterations % 100 == 0:
                print(error)
        return theta

    def cost(self) -> Cost:
        if self._parameters is None:
            raise RuntimeError('Parameters not set yet')

        regularization = parametrize(self._regularization, self._parameters.regularization_parameters)
        return make_cost(self._basic_cost, regularization)
