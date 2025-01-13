from typing import Dict, Union
import numpy as np

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.interface import VectorArray
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorSpace
from pymor.core.base import ImmutableObject
from pymor.operators.constructions import ZeroOperator

from RBInvParam.evaluators import UnAssembledA, UnAssembledB, AssembledA, AssembledB
from RBInvParam.timestepping import ImplicitEulerTimeStepper
from RBInvParam.residuals import StateResidualOperator,AdjointResidualOperator

# TODO 
# - And assert correct space
# - Add caching
# - Switch np tp pyMOR
# - Switch len(q) == 1 to q_time_dep

# Notes caching:
# - Is q t dep A(q) can be stored for all time steps or calculated on the fly.
#   - The later is maybe possible for ROMs not for FOM
# 



class InstationaryModelIP(ImmutableObject):
    def __init__(self,
                 u_0 : VectorArray, 
                 M : Operator,
                 A : Union[UnAssembledA, AssembledA],
                 L : VectorArray,
                 B : Union[UnAssembledB, AssembledB],
                 constant_cost_term: float,
                 linear_cost_term: NumpyMatrixOperator,
                 bilinear_cost_term: NumpyMatrixOperator,
                 Q : VectorSpace,
                 V : VectorSpace,
                 q_circ: VectorArray,
                 constant_reg_term: float,
                 linear_reg_term: NumpyMatrixOperator,
                 bilinear_reg_term: NumpyMatrixOperator,
                 state_residual_operator: Union[ZeroOperator, StateResidualOperator],
                 adjoint_residual_operator: Union[ZeroOperator, AdjointResidualOperator],
                 products : Dict,
                 visualizer,
                 dims: Dict,
                 model_parameter: Dict,
                 name: str = None):

        self.u_0 = u_0
        assert np.all(u_0.to_numpy() == 0)
        self.p_0 = u_0.copy()
        self.linearized_u_0 = u_0.copy()
        self.linearized_p_0 = u_0.copy()

        self.M = M 
        self.A = A 
        self.L = L 
        self.B = B 
        self.constant_cost_term = constant_cost_term 
        self.linear_cost_term = linear_cost_term 
        self.bilinear_cost_term = bilinear_cost_term 
        self.V = V
        self.Q = Q
        self.q_circ = q_circ 
        self.constant_reg_term = constant_reg_term 
        self.linear_reg_term = linear_reg_term 
        self.bilinear_reg_term = bilinear_reg_term 
        self.state_residual_operator = state_residual_operator
        self.adjoint_residual_operator = adjoint_residual_operator
        self.products = products
        self.visualizer = visualizer
        self.dims = dims
        self.model_parameter = model_parameter

        self.nt = self.dims['nt']
        self.delta_t = self.model_parameter['delta_t']
        self.q_time_dep = model_parameter['q_time_dep']

        self.timestepper = ImplicitEulerTimeStepper(
            nt = self.nt
        )

        if name:
            self.model_parameter['name'] = name

        assert self.M.source == self.M.range
        assert self.A.source == self.A.range
        assert self.M.source == self.A.source
        assert self.A.range == self.V
        assert L in self.A.range
        assert self.A.Q == self.Q
        assert self.q_circ in self.Q

        assert self.bilinear_cost_term.source == self.bilinear_cost_term.range
        assert self.bilinear_cost_term.source == self.A.range
        assert self.linear_cost_term.range == self.A.range
        assert len(self.linear_cost_term.as_range_array()) == self.dims['nt']

        assert self.bilinear_reg_term.source == self.bilinear_reg_term.range
        assert self.bilinear_reg_term.source == self.A.range
        assert self.linear_reg_term.range == self.A.range
        assert len(self.linear_reg_term.as_range_array()) == self.dims['nt']

        assert isinstance(state_residual_operator, (ZeroOperator, StateResidualOperator))
        assert isinstance(adjoint_residual_operator, (ZeroOperator, AdjointResidualOperator))

        self.state_residual_operator.source == self.A.source
        self.adjoint_residual_operator.source == self.A.source

#%% solve methods
    def solve_state(self, q: VectorArray) -> VectorArray:
        assert q in self.Q

        if self.q_time_dep:
            assert len(q) == self.dims['nt']
        else:
            assert len(q) == 1

        iterator = self.timestepper.iterate(initial_time = self.model_parameter['T_initial'], 
                                            end_time = self.model_parameter['T_final'], 
                                            initial_data = self.u_0, 
                                            q=q,
                                            operator = self.A, 
                                            rhs=self.L, 
                                            mass=self.M)
        
        u = self.V.empty(reserve= self.dims['nt'])
        for u_n, _ in iterator:
            u.append(u_n)
        return u

    def solve_adjoint(self, 
                      q: VectorArray, 
                      u: VectorArray) -> VectorArray:
        
        assert q in self.Q
        assert u in self.V

        if self.q_time_dep:
            assert len(q) == self.dims['nt']
        else:
            assert len(q) == 1

        assert len(u) == self.dims['nt']

        rhs = self.bilinear_cost_term.apply(u) - self.linear_cost_term.as_range_array()
        rhs = np.flip(rhs.to_numpy(), axis=0)
        # TODO Make def with delta_t consitent and move it the disctetirzer
        rhs = self.delta_t * self.V.make_array(rhs)

        iterator = self.timestepper.iterate(initial_time = self.model_parameter['T_initial'], 
                                            end_time = self.model_parameter['T_final'], 
                                            initial_data = self.p_0, 
                                            q=q,
                                            operator = self.A, 
                                            rhs=rhs, 
                                            mass=self.M)
        
        p = self.V.empty(reserve= self.dims['nt'])
        for p_n, _ in iterator:
            p.append(p_n)
        return self.V.make_array(np.flip(p.to_numpy(), axis=0))
    
    def solve_linearized_state(self,
                               q: VectorArray,
                               d: VectorArray,
                               u: VectorArray) -> VectorArray:
        
        assert q in self.Q
        assert d in self.Q
        assert u in self.V
        if self.q_time_dep:
            assert len(q) == self.dims['nt']
        else:
            assert len(q) == 1
        assert len(d) in [self.dims['nt'], 1]
        assert len(u) == self.dims['nt']
        
        # TODO Check if this is efficent and / or how its efficeny can be improved
        if self.q_time_dep:
            rhs = self.V.make_array(np.array([
                self.B(u[idx]).B_u(d[idx]).to_numpy()[0] for idx in range(len(u))
            ]))
        else:            
            rhs = self.V.make_array(np.array([
                self.B(u[idx]).B_u(d[0]).to_numpy()[0] for idx in range(len(u))
            ]))
        
        iterator = self.timestepper.iterate(initial_time = self.model_parameter['T_initial'], 
                                            end_time = self.model_parameter['T_final'], 
                                            initial_data = self.linearized_u_0, 
                                            q=q,
                                            operator = self.A, 
                                            rhs=rhs, 
                                            mass=self.M)        
        lin_u = self.V.empty(reserve= self.dims['nt'])
        for lin_u_n, _ in iterator:
            lin_u.append(lin_u_n)
        return lin_u
    
    def solve_linearized_adjoint(self,
                                 q: VectorArray,
                                 u: VectorArray,
                                 lin_u: VectorArray) -> VectorArray:
    
        assert q in self.Q
        assert u in self.V
        assert lin_u in self.V
        if self.q_time_dep:
            assert len(q) == self.dims['nt']
        else:
            assert len(q) == 1
        assert len(u) == self.dims['nt']
        assert len(lin_u) == self.dims['nt']

        rhs = self.bilinear_cost_term.apply(u + lin_u) - self.linear_cost_term.as_range_array()
        rhs = np.flip(rhs.to_numpy(), axis=0)
        rhs = self.delta_t * self.V.make_array(rhs)
        iterator = self.timestepper.iterate(initial_time = self.model_parameter['T_initial'], 
                                            end_time = self.model_parameter['T_final'], 
                                            initial_data = self.p_0, 
                                            q=q,
                                            operator = self.A, 
                                            rhs=rhs, 
                                            mass=self.M,
                                            )
        
        lin_p = self.V.empty(reserve= self.dims['nt'])
        for lin_p_n, _ in iterator:
            lin_p.append(lin_p_n)

        return self.V.make_array(np.flip(lin_p.to_numpy(), axis=0))
    
#%% objective and gradient
    def objective(self, 
                  u: Union[VectorArray, np.ndarray],
                  q: VectorArray = None,
                  alpha: float = 0) -> float:
        
        if q:
            assert q in self.Q
            if self.q_time_dep:
                assert len(q) == self.dims['nt']
            else:
                assert len(q) == 1

        assert len(u) == (self.dims['nt'])
        assert u in self.V
        
        # compute tracking term
        out = 0.5 * self.delta_t * np.sum(self.bilinear_cost_term.pairwise_apply2(u,u) 
                                          + (-2)  * self.linear_cost_term.as_range_array().pairwise_inner(u) 
                                          + self.constant_cost_term)
            
        if alpha > 0:
            assert q is not None
            # add regularization term if alpha >0
            return out + alpha * self.regularization_term(q)
        else:
            return out

    def gradient(self,
                 u: VectorArray,
                 p: VectorArray,
                 q: VectorArray = None,
                 alpha: float = 0) -> VectorArray:
        
        assert u in self.V
        assert p in self.V
        assert len(u) == self.dims['nt']
        assert len(p) == self.dims['nt']

        grad = np.empty((self.dims['nt'], self.dims['par_dim']))    

        # TODO Check if this is efficent and / or how its efficeny can be improved
        for idx in range(0, self.dims['nt']):
            grad[idx] = self.B(u[idx]).B_u_ad(p[idx], 'grad')

        if not self.q_time_dep:
            grad = np.sum(grad, axis=0, keepdims=True) 
        
        if alpha > 0:
            assert q is not None
            # add regularization term if alpha >0
            return self.Q.make_array(grad) + alpha * self.gradient_regularization_term(q)
        else:
            return self.Q.make_array(grad)
        
    def linearized_objective(self,
                            q: VectorArray,
                            d: VectorArray,
                            u: VectorArray,
                            lin_u: VectorArray,
                            alpha : float) -> float:

        if self.q_time_dep:
            assert len(q) == self.dims['nt']
        else:
            assert len(q) == 1
        assert len(d) in [self.dims['nt'], 1]
        assert len(u) == self.dims['nt']
        assert len(lin_u) == self.dims['nt']

        assert q in self.Q
        assert d in self.Q
        assert u in self.V
        assert lin_u in self.V

        u_q_d = u + lin_u
        out = 0.5 * self.delta_t * np.sum( \
                      self.bilinear_cost_term.pairwise_apply2(u_q_d,u_q_d) + \
                      (-2)  * self.linear_cost_term.as_range_array().pairwise_inner(u_q_d) + \
                      self.constant_cost_term)
        if alpha > 0:
            return out + alpha * self.linearized_regularization_term(q, d)
        else:
            return out
        
    def linearized_gradient(self,
                            q: VectorArray,
                            d: VectorArray,
                            u: VectorArray,
                            lin_p: VectorArray,
                            alpha : float) -> VectorArray:
        
        if self.q_time_dep:
            assert len(q) == self.dims['nt']
        else:
            assert len(q) == 1
        assert len(q) == len(d)
        assert len(u) == self.dims['nt']
        assert len(lin_p) == self.dims['nt']
        

        assert q in self.Q
        assert d in self.Q
        assert u in self.V
        assert lin_p in self.V

        grad = np.empty((self.dims['nt'], self.dims['par_dim']))
        
        # TODO Check if this is efficent and / or how its efficeny can be improved
        for idx in range(0, self.dims['nt']):
            grad[idx] = self.B(u[idx]).B_u_ad(lin_p[idx], 'grad') 

        if not self.q_time_dep:
            grad = np.sum(grad, axis=0, keepdims=True) 

        if alpha > 0:
            return self.Q.make_array(grad) + alpha * self.linarized_gradient_regularization_term(q,d)
        else:
            return self.Q.make_array(grad)
    
    def linearized_hessian(self):
        raise NotImplementedError

#%% regularization
    def regularization_term(self, 
                            q: VectorArray) -> float:
        assert q in self.Q
        if self.q_time_dep:
            assert len(q) == self.dims['nt']
        else:
            assert len(q) == 1
        
        if self.q_time_dep:
            return 0.5 * self.delta_t * np.sum(self.bilinear_reg_term.pairwise_apply2(q,q) 
                                            + (-2) * self.linear_reg_term.as_range_array().pairwise_inner(q) 
                                            + self.constant_reg_term)
        else:
            return 0.5 * self.delta_t * np.sum(self.bilinear_reg_term.pairwise_apply2(q,q)
                                            + (-2) * q.inner(self.linear_reg_term.as_range_array())
                                            + self.constant_reg_term)
            
        
    def gradient_regularization_term(self, 
                                     q: VectorArray) -> float:
        assert q in self.Q
        if self.q_time_dep:
            assert len(q) == self.dims['nt']
        else:
            assert len(q) == 1

        out = self.delta_t * (- self.linear_reg_term.as_range_array() + self.products['prod_Q'].apply(q))
        if self.q_time_dep:
            return out 
        else:
            return self.dims['nt'] * out
        
           
    def linearized_regularization_term(self, 
                                       q: VectorArray,
                                       d: VectorArray) -> float:
        if self.q_time_dep:
            assert len(q) == self.dims['nt']
        else:
            assert len(q) == 1
        assert len(q) == len(d)
        
        assert q in self.Q
        assert d in self.Q
        
        if self.q_time_dep:
            return 0.5 * self.delta_t * np.sum(self.bilinear_reg_term.pairwise_apply2(q+d,q+d)
                                            + (-2) * self.linear_reg_term.as_range_array().pairwise_inner(q+d) 
                                            + self.constant_reg_term)
        else:
            return 0.5 * self.delta_t * np.sum(self.bilinear_reg_term.pairwise_apply2(q+d,q+d)
                                            + (-2) * (q+d).inner(self.linear_reg_term.as_range_array())
                                            + self.constant_reg_term)
        
            
            
    def linarized_gradient_regularization_term(self,
                                               q: VectorArray,
                                               d: VectorArray) -> float:
        if self.q_time_dep:
            assert len(q) == self.dims['nt']
        else:
            assert len(q) == 1
        assert len(q) == len(d)
        
        assert q in self.Q
        assert d in self.Q

        out = self.delta_t * (- self.linear_reg_term.as_range_array() + self.products['prod_Q'].apply(q + d))
        if self.q_time_dep:
            return out
        else:
            return self.dims['nt'] * out

#%% error estimator

    def estimate_state_error(self,
                             u: VectorArray,
                             q: VectorArray) -> float:
        
        assert len(u) == self.nt
        

        if isinstance(self.state_residual_operator , ZeroOperator):
            r = ZeroOperator.apply(u).norm2(product=0)
        else:
            alpha_q = ...

            u_old = self.Q.zero(reserve=self.nt)
            u_old[1:] = u[1:]
            # TODO THIS IS NOT CORRECT YET. Has to be Riesz repr!
            r = self.state_residual_operator.apply(
                u = u, 
                u_old = u_old,
                q = q
            ).norm2(product=None)

        return np.sqrt(self.delta_t / alpha_q * r)


    def estimate_adjoint_error(self,
                               u: VectorArray,
                               p: VectorArray) -> float:
        pass

    def estimate_objective_error(self) -> float:
        pass

    def estimate_gradient_error(self) -> float:
        raise NotImplementedError

    def estimate_linarized_state_error(self) -> float:
        raise NotImplementedError

    def estimate_linarized_adjoint_error(self) -> float:
        raise NotImplementedError

    def estimate_linarized_objective_error(self) -> float:
        raise NotImplementedError

    def estimate_linarized_gradient_error(self) -> float:
        raise NotImplementedError

#%% compute functions                            
    def compute_objective(self, 
                          q: VectorArray,
                          alpha : float = 0) -> float:

        u = self.solve_state(q)
        return self.objective(u, q, alpha)
    
    def compute_gradient(self,
                         q: VectorArray,
                         alpha : float = 0) -> float:
        
        u = self.solve_state(q)
        p = self.solve_adjoint(q, u)
        return self.gradient(u, p, alpha)
        

    def compute_linearized_objective(self,
                                     q: VectorArray,
                                     d: VectorArray,
                                     alpha : float) -> float:

        u = self.solve_state(q)
        lin_u = self.solve_linearized_state(q, d, u)
        return self.linearized_objective(q, d, u, lin_u, alpha)

    def compute_linearized_gradient(self,
                                    q: VectorArray,
                                    d: VectorArray,
                                    alpha : float) -> float:

        u = self.solve_state(q)
        lin_u = self.solve_linearized_state(q, d, u)
        lin_p = self.solve_linearized_adjoint(q, u, lin_u)

        return self.linearized_gradient(q, d, u, lin_p, alpha)
 
#%% helpers
    def pymor_to_numpy(self,q):
        return q.to_numpy()
    
    def numpy_to_pymor(self,q):
        return self.Q.make_array(q)
