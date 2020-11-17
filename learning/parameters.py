from typing import NamedTuple, Optional, Tuple

from algebra import Vector
from cost import BasicCost, Regularization
from cost.cost import make_cost
from data.basis_functions import BasisFunction


class Parameters(NamedTuple):
    basic_cost: BasicCost
    regularization: Regularization
    gradient_step: float
    init_theta: Optional[Vector]
    basis_functions: Tuple[BasisFunction, ...]

    def cost(self):
        return make_cost(self.basic_cost, self.regularization)

    def __repr__(self):
        return str({
            'Basic cost'     : self.basic_cost.__name__,
            'Regularization' : f'{self.regularization.__regularization__.__name__} '
                               f'with parameters {self.regularization.__parameters__}',
            'Gradient step'  : f'{self.gradient_step:.6f}',
            'Initial theta'  : str(self.init_theta) if self.init_theta else 'Zeros',
            'Basis functions': self.basis_functions.__description__
        })
