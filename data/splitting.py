from random import shuffle
from typing import Tuple

from algebra import Matrix, Vector


def split2(x: Matrix, y: Vector, p: float) -> Tuple[Tuple[Matrix, Vector], Tuple[Matrix, Vector]]:
    assert len(x) == len(y) and 0. <= p <= 1.
    domain = list(zip(x, y))
    shuffle(domain)

    prefix_size = int(p * len(domain))
    prefix, suffix = domain[:prefix_size], domain[prefix_size:]

    first, second = tuple(zip(*prefix)), tuple(zip(*suffix))
    if len(first) == 0:
        first = (tuple(), tuple())
    if len(second) == 0:
        second = (tuple(), tuple())

    return first, second


def split3(x: Matrix, y: Vector, p1: float, p2: float) -> \
        Tuple[Tuple[Matrix, Vector], Tuple[Matrix, Vector], Tuple[Matrix, Vector]]:
    (x1, y1), (x2, y2) = split2(x, y, p1)
    (x3, y3), (x4, y4) = split2(x2, y2, p2 / (1 - p1))
    return (x1, y1), (x3, y3), (x4, y4)
