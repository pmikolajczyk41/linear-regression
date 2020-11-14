from enum import Enum
from statistics import mean, stdev

from data import Vector, vector


def _min_max_1(data: Vector) -> Vector:
    minim, maxim = min(data), max(data)
    return vector(map(lambda x: (x - minim) / (maxim - minim), data))


def _min_max_2(data: Vector) -> Vector:
    avg, minim, maxim = mean(data), min(data), max(data)
    return vector(map(lambda x: (x - avg) / (maxim - minim), data))


def _standard(data: Vector) -> Vector:
    avg, sdev = mean(data), stdev(data)
    return vector(map(lambda x: (x - avg) / sdev, data))


class ScalingType(Enum):
    MIN_MAX_1 = _min_max_1
    MIN_MAX_2 = _min_max_2
    STANDARD = _standard


def normalize(data: Vector, scaling_type: ScalingType) -> Vector:
    return scaling_type.value(data)
