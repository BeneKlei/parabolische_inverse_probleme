from typing import Dict, Union, List
import scipy
import copy

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray, VectorSpace
from pymor.operators.constructions import InverseOperator

from RBInvParam.evaluators import EvaluatorA


class TimeResidualOperator(Operator):
    def __init__(self,
                 M : Operator,
                 A : EvaluatorA,
                 Q : VectorSpace,
                 V : VectorSpace,
                 products : Dict,
                 setup: Dict,
                 bases: Dict,
                 riesz_representative : bool = False,
                 gram_operator: Operator = None,
                 zeta: float = 1.0):
        
        self.products = products
        self.setup = setup
        self.riesz_representative = riesz_representative

        assert (0.0 <= zeta) and (zeta <= 1.0)
        self.zeta = zeta

        self.delta_t = self.setup['delta_t']
        assert self.delta_t > 0
        self.rez_delta_t = 1 / self.delta_t
        self.q_time_dep = self.setup['q_time_dep']
        self.nt = self.setup['dims']['nt']
        
        assert not M.parametric
        self.M = M.assemble() 
        self.A = A
    
        self.V = V
        self.Q = Q

        self.source = A.source
        self.range = A.range
        self.bases = bases
    
        assert self.M.source == self.A.source
        assert self.A.source == self.V
        assert self.A.Q == self.Q 

        assert 'parameter_basis' in self.bases.keys()
        assert 'state_basis' in self.bases.keys()
        if len(self.bases['parameter_basis']) != 0:
            assert self.bases['parameter_basis'] in self.Q
        if len(self.bases['state_basis']) != 0:
            assert self.bases['state_basis'] in self.V
        
        # TODO Maybe set solver options globally
        self.gram_operator = gram_operator
            
        if riesz_representative:
            assert gram_operator is not None
            self.riesz_op = InverseOperator(self.gram_operator)
    
    def _precompute_residual_A_q(self, 
                                 q: VectorArray) -> List:
        if len(self.bases['parameter_basis']) != 0:
            q = self._reconstruct(q, basis='parameter_basis')
        assert q in self.Q

        return self.A(q).assemble()
        
    def _reconstruct(self, u, basis='state_basis'):
        return self.bases[basis][:u.dim].lincomb(u.to_numpy())

    def _apply(self,
               rhs: VectorArray, 
               u: VectorArray,
               mass_u: VectorArray,
               q: VectorArray,
               use_cached_operators: bool = False,
               cached_operators: Dict = None) -> VectorArray:

        if use_cached_operators:
            assert cached_operators
            'q' in cached_operators.keys()
            'residual_A_q' in cached_operators.keys()

            if len(cached_operators['q']) > 0:
                assert ((cached_operators['q']-q).norm() <= 1e-16)[0]

            if self.setup['q_time_dep']:
                assert len(cached_operators['residual_A_q']) == (self.nt + 1)
            else:
                assert len(cached_operators['residual_A_q']) == 1
        
        if len(self.bases['parameter_basis']) != 0:
            q = self._reconstruct(q, basis='parameter_basis')
        if len(self.bases['state_basis']) != 0:
            u = self._reconstruct(u, basis='state_basis')
            mass_u = self._reconstruct(mass_u, basis='state_basis')

        assert q in self.Q
        assert rhs in self.A.range
        assert u in self.V
        assert mass_u in self.V

        assert len(u) == len(mass_u)
        assert len(u) == (self.nt + 1)

        if not self.q_time_dep:
            assert len(q) == 1
        else:
            assert len(q) == len(u)

        if use_cached_operators:
            A_q = cached_operators['residual_A_q']
            if self.q_time_dep:
                Au = self.A.range.empty(reserve = len(u)) 
                for i in range(len(u)):
                    Au.append(A_q[i].apply(u[i]))
            else:
                Au = A_q[0].apply(u)
        else:
            if self.q_time_dep:
                Au = self.A.range.empty(reserve = len(u)) 
                for i in range(len(u)):
                    Au.append(self.A(q[i]).apply(u[i]))
            else:
                Au = self.A(q[0]).apply(u)

        print(Au)

        Mmass_u = self.M.apply(mass_u)
        _Mmass_u = Au.space.zeros(len(Au) - 1)
        _Mmass_u.axpy(1.0, Mmass_u[1:])
        _Mmass_u.axpy(-1.0, Mmass_u[:-1])
        _Mmass_u.scal(self.rez_delta_t)

        if self.zeta != 1.0:
            _Au = Au.space.zeros(len(Au) - 1)
            _Au.axpy(self.zeta, Au[1:])
            _Au.axpy(1 - self.zeta, Au[:-1])
        else:
            _Au = Au

        if self.zeta != 1.0:
            _rhs = rhs.space.zeros(len(rhs) - 1)
            _rhs.axpy(self.zeta, rhs[1:])
            _rhs.axpy(1 - self.zeta, rhs[:-1])
        else: 
            _rhs = rhs
        
        #assert len(R) == self.nt
        R = - _Au - _Mmass_u + _rhs

        assert len(R) == self.nt

        if self.riesz_representative:
            R = self.riesz_op.apply(R)
            return R
        else:
            return R
        
class StateResidualOperator(TimeResidualOperator):
    def __init__(self,
                 M : Operator,
                 A : EvaluatorA,
                 L : VectorArray,
                 Q : VectorSpace,
                 V : VectorSpace,
                 products : Dict,
                 setup: Dict,
                 bases: Dict,
                 zeta: float,
                 riesz_representative : bool = False,
                 gram_operator: Operator = None):
        
        super().__init__(M = M,
                         A = A,
                         Q = Q,
                         V = V,
                         products = products,
                         setup = setup,
                         bases=bases,
                         riesz_representative = riesz_representative,
                         gram_operator = gram_operator,
                         zeta=zeta)
        
        self.L = L
        assert self.L in self.M.range
        assert isinstance(self.L, VectorArray)
        assert len(self.L) in [1, self.nt + 1]

    def apply(self,
              u: VectorArray,
              mass_u: VectorArray,
              q: VectorArray,
              use_cached_operators: bool = False,
              cached_operators: Dict = None) -> VectorArray:
        
        assert len(u) == len(mass_u) == (self.nt + 1)
        return self._apply(rhs = self.L,
                           u = u,
                           mass_u = mass_u,
                           q=q,
                           use_cached_operators=use_cached_operators,
                           cached_operators=cached_operators)
     
class AdjointResidualOperator(TimeResidualOperator):
    def __init__(self,
                 M : Operator,
                 A : EvaluatorA,
                 linear_cost_term: NumpyMatrixOperator,
                 bilinear_cost_term: NumpyMatrixOperator,
                 Q : VectorSpace,
                 V : VectorSpace,
                 products : Dict,
                 setup : Dict,
                 bases: Dict,
                 riesz_representative : bool = False,
                 gram_operator: Operator = None):
        
        super().__init__(M = M,
                         A = A, # A is symmetric
                         Q = Q,
                         V = V,                         
                         products = products,
                         setup = setup,
                         bases=bases,
                         riesz_representative=riesz_representative,
                         gram_operator=gram_operator)
            
        self.bilinear_cost_term = bilinear_cost_term
        self.linear_cost_term = linear_cost_term

        assert self.bilinear_cost_term.range == self.A.range
        assert self.linear_cost_term in self.A.range
        assert len(self.linear_cost_term) == (self.setup['dims']['nt'] + 1)
        
    def apply(self,
              p: VectorArray,
              mass_p: VectorArray,
              u: VectorArray,
              q: VectorArray,
              use_cached_operators: bool = False,
              cached_operators: Dict = None) -> VectorArray:
        
        assert len(p) == len(mass_p) == len(u) == (self.nt + 1)
        
        rhs = self.bilinear_cost_term.apply(u) - self.linear_cost_term
        rhs *= self.delta_t
        rhs *= (-1) 

        return self._apply(rhs = rhs,
                           u = p,
                           u_old = mass_p,
                           q=q,
                           use_cached_operators=use_cached_operators,
                           cached_operators=cached_operators)
