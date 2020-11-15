from itertools import repeat
from pathlib import Path
from statistics import mean

from algebra import Vector, vector, mult_vs, sum_vv
from cost import Cost
from cost.basic_cost import squared_error
from cost.cost import make_cost
from cost.regularization import ridge, parametrize
from data.X import X
from data.loading import load
from data.normalization import ScalingType
from learning.stop import StopCondition


class LinearLearner:
    @staticmethod
    def learn(x: X,
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
            print(error)
            if stop:
                break
        print('Done')
        return theta


if __name__ == '__main__':
    x, y = load(Path('../noise.data'))
    x = X(x).normalize(ScalingType.MIN_MAX_1)
    regularization = parametrize(ridge, 0.4)
    cost = make_cost(squared_error, regularization)
    stop_condition = StopCondition(None, None, 10000)
    LinearLearner().learn(x, y, cost, stop_condition, 0.1)
