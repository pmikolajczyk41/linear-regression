from typing import Callable, Tuple

from algebra import Vector, Scalar
from data.X import X

BasicCost = Cost = Callable[[X, Vector, Vector], Tuple[Scalar, Vector]]
Regularization = Callable[[Vector], Tuple[Scalar, Vector]]
