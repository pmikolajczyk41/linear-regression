from typing import NamedTuple, Optional, Tuple, Any

from algebra import Vector


class Parameters(NamedTuple):
    regularization_parameters: Tuple[Any, ...]
    gradient_step: float
    init_theta: Optional[Vector] = None

    def __repr__(self):
        return str({
            'Regularization parameters': f'{self.regularization_parameters}',
            'Gradient step'            : f'{self.gradient_step:.6f}',
            'Initial theta'            : str(self.init_theta) if self.init_theta else 'Zeros',
        })
