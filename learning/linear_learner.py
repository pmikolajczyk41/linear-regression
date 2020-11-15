from cost import Cost
from data import Vector
from data.X import X
from learning.stop import StopCondition


class LinearLearner:
    def learn(self, x: X, y: Vector, cost: Cost, stop_condition: StopCondition) -> Vector:
        pass
