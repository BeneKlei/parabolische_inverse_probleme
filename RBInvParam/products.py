import numpy as np

from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray, VectorSpace 

class BochnerProductOperator(Operator):
    def __init__(self, 
                 product: Operator,
                 delta_t: float,
                 space : VectorSpace,
                 nt : int):
        
        self.product = product
        self.delta_t = delta_t
        self.space = space
        self.nt = nt

        assert self.product.source == self.product.range
        assert self.product.source == self.space
    
    def apply(self, U, mu=None):
        raise NotImplementedError

    # TODO Rename to pairwise_apply2
    def apply2(self, 
               V: VectorArray, 
               U: VectorArray,
               mu=None) -> float:
        
        assert V in self.space
        assert U in self.space
        assert len(V) == len(U) == self.nt

        return np.array([np.sum(
            self.delta_t * \
            self.product.pairwise_apply2(V,U, mu),
            keepdims=True
        )])