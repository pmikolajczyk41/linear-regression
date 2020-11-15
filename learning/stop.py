import sys
from itertools import cycle
from operator import sub
from typing import Optional, Tuple

from algebra import Vector


class StopCondition:
    def __init__(self, gradient_eps: Optional[float], error_eps: Optional[float], iterations: Optional[int]):
        self._gradient_eps = gradient_eps or -1.
        self._error_eps = error_eps or -1.
        self._iterations = iterations or sys.maxsize

        self._last_gradient = cycle((0.,))
        self._last_error = 0.

    def _set_memory(self, last_gradient, last_error):
        self._last_gradient = last_gradient
        self._last_error = last_error

    def update(self, gradient: Vector, error: float, iterated: int = 1) -> Tuple[Optional['StopCondition'], bool]:
        grad_change = max(map(lambda x: abs(sub(*x)), zip(gradient, self._last_gradient)))
        error_change = abs(error - self._last_error)
        iterations_left = self._iterations - iterated

        stop = (grad_change < self._gradient_eps or
                error_change < self._error_eps or
                iterations_left <= 0)

        if stop:
            return None, True

        new_condition = StopCondition(self._gradient_eps, self._error_eps, iterations_left)
        new_condition._set_memory(gradient, error)
        return new_condition, False
