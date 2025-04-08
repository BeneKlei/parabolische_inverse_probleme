from typing import Dict, Union, List
import scipy
import copy

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray, VectorSpace
from pymor.operators.constructions import InverseOperator

from RBInvParam.evaluators import UnAssembledA, UnAssembledB, AssembledA, AssembledB

import timeit

class ImplicitEulerResidualOperator(Operator):
    def __init__(self,
                 M : Operator,
                 A : Union[UnAssembledA, AssembledA],
                 Q : VectorSpace,
                 V : VectorSpace,
                 riesz_representative : bool,
                 products : Dict,
                 setup: Dict,
                 bases: Dict):
        
        self.products = products
        self.setup = setup
        self.riesz_representative = riesz_representative

        if riesz_representative:
            assert 'prod_V' in self.products
            self.products['prod_V'].range == V

        self.delta_t = self.setup['model_parameter']['delta_t']
        self.q_time_dep = self.setup['model_parameter']['q_time_dep']
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
        
        if riesz_representative:
            # TODO Maybe set solver options globally
            self.riesz_op = copy.deepcopy(self.products['prod_V'])
            self.riesz_op = InverseOperator(self.riesz_op)


            #self.riesz_op.with_(solver_options='scipy_spsolve')
        
        self._cached_d = None
        self._cached_residual_A_d = []
    
    def _compute_cached_residual_A_d(self,
                                     d: VectorArray) -> None:
        
        if self._cached_d == d:
            return
        
        self._cached_d = d
        self._cached_residual_A_d = []
        
        if self.setup['model_parameter']['q_time_dep']:
            assert len(self._cached_d) == self.nt
            for n in range(self.nt):
                self._cached_residual_A_d.append(
                    self._precompute_residual_A_q(d[n])-self.A.constant_operator
                )
        else:
            assert len(self._cached_d) == 1
            self._cached_residual_A_d.append(
                self._precompute_residual_A_q(d[0])-self.A.constant_operator
            )

    
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
               u_old: VectorArray,
               q: VectorArray,
               t: float = 0.0,
               d: VectorArray = None,
               use_cached_operators: bool = False,
               cached_operators: Dict = None) -> VectorArray:

        start = timeit.default_timer()
        if use_cached_operators:
            assert cached_operators
            'q' in cached_operators.keys()
            'residual_A_q' in cached_operators.keys()

            # if len(cached_operators['q']) > 0:
            #     assert ((cached_operators['q']-q).norm() <= 1e-16)[0]

            if self.setup['model_parameter']['q_time_dep']:
                assert len(cached_operators['residual_A_q']) == self.nt
            else:
                assert len(cached_operators['residual_A_q']) == 1

            if d:
                self._compute_cached_residual_A_d(d)

        
        if len(self.bases['parameter_basis']) != 0:
            q = self._reconstruct(q, basis='parameter_basis')
        if len(self.bases['state_basis']) != 0:
            u = self._reconstruct(u, basis='state_basis')
            u_old = self._reconstruct(u_old, basis='state_basis')

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

        if use_cached_operators:
            A_q = cached_operators['residual_A_q']
            if self.q_time_dep:
                Au = self.A.range.empty(reserve = len(u)) 
                if d:
                    A_ = [
                        A_q[i] + t * (self._cached_residual_A_d[i])    
                        for i in range(len(self._cached_residual_A_d))
                    ]
                else:
                    A_ = A_q
                for i in range(len(u)):
                    Au.append(A_[i].apply(u[i]))

                # for i in range(len(u)):
                #     A_ = A_q[i]
                #     if d:
                #         A_ += t * (self._cached_residual_A_d[i])
                    
                #     Au.append(A_.apply(u[i]))
            else:
                A_ = A_q[0]
                if d:
                    A_ += t * self._cached_residual_A_d[0]
                Au = A_.apply(u)
            
            # if d:
            #     A_q = cached_operators['residual_A_q'] 
            #     print(type(self.A))
            #     print(self.A.constant_operator)
            #     print(self.A(q[0]).assemble().matrix.todense())
            #     print(self._cached_residual_A_d[0])
            #     print((A_q[0] + t * (self._cached_residual_A_d[0]-self.A.constant_operator)).assemble().matrix.todense())

        else:
            if self.q_time_dep:
                Au = self.A.range.empty(reserve = len(u)) 
                for i in range(len(u)):
                    Au.append(self.A(q[i]).apply(u[i]))
            else:
                Au = self.A(q[0]).apply(u)

            A_q = cached_operators['residual_A_q']
            # print("§§§§§§§§§§§§§§§§§§§§")
            # print(self.A(q[0]).assemble().matrix.todense())
        
        # print(self._cached_residual_A_d[0])
        # print((A_q[0] + t * self._cached_residual_A_d[0]).assemble().matrix.todense())
        
        R = - Au - 1/ self.delta_t * self.M.apply(u - u_old) + rhs
        stop = timeit.default_timer()

        print('Time: ', stop - start)  

        if self.riesz_representative:
            R = self.riesz_op.apply(R)
            return R
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
                 setup: Dict,
                 bases: Dict):
        
        super().__init__(M = M,
                         A = A,
                         Q = Q,
                         V = V,
                         riesz_representative = riesz_representative,
                         products = products,
                         setup = setup,
                         bases=bases)
        
        self.L = L
        assert self.L in self.M.range
        assert isinstance(self.L, VectorArray)
        assert len(self.L) in [1, self.nt]

    # TODO Add asserts everywhere for d
    def apply(self,
              u: VectorArray,
              u_old: VectorArray,
              q: VectorArray,
              t: float = 0.0,
              d: VectorArray = None,
              use_cached_operators: bool = False,
              cached_operators: Dict = None) -> VectorArray:
        
        assert len(u) == len(u_old) == self.nt
        return self._apply(rhs = self.L,
                           u = u,
                           u_old = u_old,
                           q=q,
                           t=t,
                           d=d,
                           use_cached_operators=use_cached_operators,
                           cached_operators=cached_operators)

        
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
                 setup : Dict,
                 bases: Dict):
        
        super().__init__(M = M,
                         A = A, # A is symmetric
                         Q = Q,
                         V = V,
                         riesz_representative = riesz_representative,
                         products = products,
                         setup = setup,
                         bases=bases)
            
        self.bilinear_cost_term = bilinear_cost_term
        self.linear_cost_term = linear_cost_term

        #assert self.bilinear_cost_term.source == self.bilinear_cost_term.range
        #assert self.bilinear_cost_term.source == self.A.source
        assert self.bilinear_cost_term.range == self.A.range
        assert self.linear_cost_term.range == self.A.range
        assert len(self.linear_cost_term.as_range_array()) == self.setup['dims']['nt']
        

    def apply(self,
              p: VectorArray,
              p_old: VectorArray,
              u: VectorArray,
              q: VectorArray,
              t: float = 0.0,
              d: VectorArray = None,
              use_cached_operators: bool = False,
              cached_operators: Dict = None) -> VectorArray:
        
        assert len(p) == len(p_old) == len(u) == self.nt
        
        rhs = self.bilinear_cost_term.apply(u) - self.linear_cost_term.as_range_array()
        rhs *= self.delta_t

        return self._apply(rhs = rhs,
                           u = p,
                           u_old = p_old,
                           q=q,
                           t=t,
                           d=d,
                           use_cached_operators=use_cached_operators,
                           cached_operators=cached_operators)
