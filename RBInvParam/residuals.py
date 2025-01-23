from typing import Dict, Union

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray, VectorSpace

from RBInvParam.evaluators import UnAssembledA, UnAssembledB, AssembledA, AssembledB

# Assembly ResidualOps
# - Assemble it first naively, i.e. do not use projection to an (approximate) image basis
#   - Constructing this basis can be to costly
#   - Rememeber that the error is not used solving the linear problem 
#   - But only in backtracking from the obtained direction 


class ImplicitEulerResidualOperator(Operator):
    def __init__(self,
                 M : Operator,
                 A : Union[UnAssembledA, AssembledA],
                 Q : VectorSpace,
                 V : VectorSpace,
                 riesz_representative : bool,
                 products : Dict,
                 setup: Dict):
        
        self.products = products
        self.setup = setup
        self.riesz_representative = riesz_representative

        if riesz_representative:
            assert 'prod_V' in self.products
            self.products['prod_V'].range == V

        self.delta_t = self.setup['model_parameter']['delta_t']
        self.q_time_dep = self.setup['model_parameter']['q_time_dep']
        self.nt = self.setup['dims']['nt']
        
        self.M = M
        self.A = A
    
        self.V = V
        self.Q = Q

        self.source = A.source
        self.range = A.range
        
        # assert self.M.source == self.M.range
        # assert self.A.source == self.A.range
        assert self.M.source == self.A.source
        assert self.A.source == self.V
        assert self.A.Q == self.Q        
    
    def _apply(self,
               rhs: VectorArray, 
               u: VectorArray,
               u_old: VectorArray,
               q: VectorArray) -> VectorArray:
        
        assert q in self.Q
        assert rhs in self.A.range
        assert u in self.V
        assert u_old in self.V

        assert len(u) == len(u_old)
        assert len(u) <= self.nt

        if not self.q_time_dep:
            assert len(q) == 1
        else:
            assert len(q) == len(u)

        if self.q_time_dep:
            Au = self.A.range.empty(reserve = len(u)) 
            for i in range(len(u)):
                Au.append(self.A(q[i]).apply(u[i]))
        else:
            Au = self.A(q[0]).apply(u)

        R = - Au - 1/ self.delta_t * self.M.apply(u - u_old) + rhs
        
        if self.riesz_representative:
            return self.products['prod_V'].apply_inverse(R)
        else:
            return R
        

class StateResidualOperator(ImplicitEulerResidualOperator):
    def __init__(self,
                 M : Operator,
                 A : Union[UnAssembledA, AssembledA],
                 L : VectorArray,
                 Q : VectorSpace,
                 V : VectorSpace,
                 riesz_representative : bool,
                 products : Dict,
                 setup: Dict):
        
        super().__init__(M = M,
                         A = A,
                         Q = Q,
                         V = V,
                         riesz_representative = riesz_representative,
                         products = products,
                         setup = setup)
        
        self.L = L
        assert self.L in self.M.range
        assert isinstance(self.L, VectorArray)
        assert len(self.L) in [1, self.nt]

    def apply(self,
              u: VectorArray,
              u_old: VectorArray,
              q: VectorArray) -> VectorArray:
        
        assert len(u) == len(u_old) == self.nt
        return self._apply(rhs = self.L,
                           u = u,
                           u_old = u_old,
                           q=q)

        
class AdjointResidualOperator(ImplicitEulerResidualOperator):
    def __init__(self,
                 M : Operator,
                 A : Union[UnAssembledA, AssembledA],
                 linear_cost_term: NumpyMatrixOperator,
                 bilinear_cost_term: NumpyMatrixOperator,
                 Q : VectorSpace,
                 V : VectorSpace,
                 riesz_representative : bool,
                 products : Dict,
                 setup : Dict):
        
        super().__init__(M = M,
                         A = A, # A is symmetric
                         Q = Q,
                         V = V,
                         riesz_representative = riesz_representative,
                         products = products,
                         setup = setup)
            
        self.bilinear_cost_term = bilinear_cost_term
        self.linear_cost_term = linear_cost_term

        #assert self.bilinear_cost_term.source == self.bilinear_cost_term.range
        assert self.bilinear_cost_term.source == self.A.source
        assert self.bilinear_cost_term.range == self.A.range
        assert self.linear_cost_term.range == self.A.range
        assert len(self.linear_cost_term.as_range_array()) == self.setup['dims']['nt']
        

    def apply(self,
              p: VectorArray,
              p_old: VectorArray,
              u: VectorArray,
              q: VectorArray) -> VectorArray:
        
        assert len(p) == len(p_old) == len(u) == self.nt
        
        rhs = self.bilinear_cost_term.apply(u) - self.linear_cost_term.as_range_array()
        rhs *= self.delta_t

        return self._apply(rhs = rhs,
                           u = p,
                           u_old = p_old,
                           q=q)
