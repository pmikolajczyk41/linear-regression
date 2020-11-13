from enum import Enum, auto
from statistics import mean, stdev
from typing import Tuple


class ScalingType(Enum):
    MIN_MAX_1 = auto()
    MIN_MAX_2 = auto()
    STANDARD = auto()


def normalize(data: Tuple[float], type: ScalingType) -> Tuple[float]:
    if type == ScalingType.MIN_MAX_1:
        return _min_max_1(data)
    elif type == ScalingType.MIN_MAX_2:
        return _min_max_2(data)
    elif type == ScalingType.STANDARD:
        return _standard(data)
    assert False, 'Unknown scaling type'


def _min_max_1(data: Tuple[float]) -> Tuple[float]:
    minim, maxim = min(data), max(data)
    return tuple(map(lambda x: float((x - minim) / (maxim - minim)), data))


def _min_max_2(data: Tuple[float]) -> Tuple[float]:
    avg, minim, maxim = mean(data), min(data), max(data)
    return tuple(map(lambda x: float((x - avg) / (maxim - minim)), data))


def _standard(data: Tuple[float]) -> Tuple[float]:
    avg, sdev = mean(data), stdev(data)
    return tuple(map(lambda x: float((x - avg) / sdev), data))
