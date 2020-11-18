from copy import copy
from functools import partial
from random import gauss
from statistics import mean
from typing import NoReturn, Iterable, List, Tuple

from algebra import Vector, vector, mult_vs, sum_vv, mult_mv
from cost import BasicCost, UnparameterizedRegularization, Cost
from cost.cost import make_cost
from cost.regularization import parametrize
from data.X import X
from data.basis_functions import BasisFunctions
from learning.parameters import Parameters
from learning.stop import StopCondition


class Model:
    stop_condition = StopCondition(None, 0.000001, 500)

    def __init__(self, basic_cost: BasicCost,
                 regularization: UnparameterizedRegularization,
                 basis_functions: BasisFunctions):
        self._basic_cost = basic_cost
        self._regularization = regularization
        self._basis_functions = basis_functions

        self._parameters = None

    def evaluate_parameters(self,
                            x_train: X, y_train: Vector, x_val: X, y_val: Vector,
                            parameters: Iterable[Parameters]) -> List[Tuple[float, Vector, Parameters]]:
        x_train = x_train.convert(self._basis_functions)
        x_val = x_val.convert(self._basis_functions).append_ones()

        cost = lambda ps: make_cost(self._basic_cost, parametrize(self._regularization, ps))

        train = partial(self._train, x_train, y_train)
        trained = map(lambda p: train(cost(p.regularization_parameters), p.gradient_step, p.stdev), parameters)

        def evaluate(paired):
            theta, params = paired
            cum_error, _ = self._basic_cost(x_val, theta, y_val)
            error = cum_error / x_val.nsamples()
            return error, theta, params

        evaluated = list(map(lambda paired: evaluate(paired), zip(trained, parameters)))
        return evaluated

    def set_parameters(self, parameters: Parameters) -> NoReturn:
        self._parameters = parameters

    def set_stop_condition(self, stop_condition: StopCondition) -> NoReturn:
        self.stop_condition = stop_condition

    def train(self, x: X, y: Vector) -> Vector:
        if self._parameters is None:
            raise RuntimeError('Parameters not set yet')

        x = x.convert(self._basis_functions)
        reg = parametrize(self._regularization, self._parameters.regularization_parameters)
        cost = make_cost(self._basic_cost, reg)

        self._theta = self._train(x, y, cost, self._parameters.gradient_step, self._parameters.stdev)
        return copy(self._theta)

    def predict(self, x: X) -> Vector:
        if self._theta is None:
            raise RuntimeError('Not trained yet')
        x = x.convert(self._basis_functions).append_ones()
        return mult_mv(x.by_sample(), self._theta)

    def error(self, x: X, y: Vector) -> float:
        if self._theta is None:
            raise RuntimeError('Not trained yet')
        x = x.convert(self._basis_functions).append_ones()
        error, _ = self._basic_cost(x, self._theta, y)
        return error / len(y)

    def _train(self, x: X, y: Vector, cost: Cost, step: float, stdev: float) -> Vector:
        x = x.append_ones()
        m = x.nsamples()

        theta = vector(map(lambda _: gauss(0., stdev), range(x.nfeatures() - 1))) + (gauss(mean(y), stdev),)

        stop_condition = self.stop_condition
        while True:
            error, gradient = cost(x, theta, y)
            error, gradient = error / m, mult_vs(gradient, 1 / m)

            theta = sum_vv(theta, mult_vs(gradient, -step))

            stop_condition, stop = stop_condition.update(gradient, error)
            if stop:
                break
        return theta

    def cost(self) -> BasicCost:
        return self._basic_cost

    def __repr__(self):
        props = {
            'Basic cost function': self._basic_cost.__name__,
            'Regularization'     : self._regularization.__name__,
            'Basis functions'    : self._basis_functions.__description__,
        }
        if self._parameters is not None:
            props['Gradient step'] = str(self._parameters.gradient_step),
            props['Regularization parameters'] = str(self._parameters.regularization_parameters)
            props['Standard deviation'] = str(self._parameters.stdev)

        return str(props)
