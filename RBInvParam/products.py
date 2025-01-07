import numpy as np

from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray

class BochnerProductOperator(Operator):
    def __init__(self, 
                 product: Operator,
                 delta_t: float):
        
        self.product = product
        self.delta_t = delta_t

    def apply(self, U, mu=None):
        raise NotImplementedError

    def apply2(self, 
               V: VectorArray, 
               U: VectorArray,
               mu=None) -> float:
        
        return np.sum(
            self.delta_t * \
            self.product.pairwise_apply2(V,U, mu)
        )