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
        assert len(V) == len(U) == (self.nt + 1)

        return np.array([np.sum(
            self.delta_t * \
            self.product.pairwise_apply2(V,U, mu),
            keepdims=True
        )])
    

class EnergyProductOperator(Operator):
    def __init__(self, 
                 kinetic_product: Operator,
                 potential_product: Operator,
                 space : VectorSpace):
                 
    
        self.kinetic_product = kinetic_product
        self.potential_product = potential_product
        
        self.space = space
        self.source = space
        self.range = space

        assert self.kinetic_product.source == self.kinetic_product.range        
        assert self.kinetic_product.source == self.space
        assert self.potential_product.source == self.potential_product.range        
        assert self.potential_product.source == self.space
    
    def apply(self,
              U : VectorArray,
              mu=None) -> float:

        assert U in self.space
        return self.kinetic_product.apply(U) + self.potential_product.apply(U)

    def apply2(self, 
               V: VectorArray, 
               U: VectorArray,
               mu=None) -> float:

        assert V in self.space
        assert U in self.space

        return self.kinetic_product.apply2(V,U) + self.potential_product.apply2(V,U)
