from itertools import repeat
from statistics import mean

from algebra import Vector, vector, mult_vs, sum_vv
from cost import Cost
from data.X import X
from learning.parameters import Parameters
from learning.stop import StopCondition


class LinearLearner:
    def train(self, x: X, y: Vector, stop_condition: StopCondition, params: Parameters):
        x = x.convert(params.basis_functions)
        cost = params.cost()

        theta = self._train(x, y, cost, stop_condition, params.gradient_step, params.init_theta)
        return theta

    @staticmethod
    def _train(x: X,
               y: Vector,
               cost: Cost,
               stop_condition: StopCondition,
               step: float,
               init_theta: Vector = None) -> Vector:
        x = x.append_ones()
        m = x.nsamples()

        if init_theta is None:
            theta = vector(repeat(0., x.nfeatures() - 1)) + (mean(y),)
        else:
            assert len(init_theta) == x.nfeatures()
            theta = init_theta

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
