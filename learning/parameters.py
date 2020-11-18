from typing import NamedTuple, Tuple, Any


class Parameters(NamedTuple):
    regularization_parameters: Tuple[Any, ...]
    gradient_step: float
    stdev: float

    def __repr__(self):
        return str({
            'Regularization parameters': f'{self.regularization_parameters}',
            'Gradient step'            : f'{self.gradient_step:.6f}',
            'Standard deviation'       : f'{self.stdev:.2f}',
        })
