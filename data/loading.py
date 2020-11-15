from pathlib import Path
from typing import Tuple

from algebra import Matrix, Vector, Scalar


def load(filepath: Path) -> Tuple[Matrix, Vector]:
    assert filepath.exists() and filepath.is_file()

    with filepath.open() as file:
        lines = file.readlines()

    samples = map(lambda l: l.split(), lines)
    samples = (tuple(map(lambda v: Scalar(v), s)) for s in samples)
    samples = ((s[:-1], s[-1]) for s in samples)

    return tuple(zip(*samples))
