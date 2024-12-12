from typing import Dict, Union
import numpy as np

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.interface import VectorArray
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorSpace

from evaluators import UnAssembledA, UnAssembledB, AssembledA, AssembledB
from timestepping import ImplicitEulerTimeStepper

# TODO 
# - And assert correct space
# - Add caching

# Notes caching:
# - Is q t dep A(q) can be stored for all time steps or calculated on the fly.
#   - The later is maybe possible for ROMs not for FOM
# 

# TODO 
# Switch np tp pyMOR

class InstationaryModelIP:
    def __init__(self,
                 u_0 : VectorArray, 
                 M : Operator,
                 A : Union[UnAssembledA, AssembledA],
                 f : VectorArray,
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
                 products : Dict,
                 visualizer,
                 dims: Dict,
                 model_parameter: Dict):

        self.u_0 = u_0
        assert np.all(u_0.to_numpy() == 0)
        self.p_0 = u_0.copy()
        self.linearized_u_0 = u_0.copy()
        self.linearized_p_0 = u_0.copy()


        self.M = M 
        self.A = A 
        self.f = f 
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
        self.products = products
        self.visualizer = visualizer
        self.dims = dims
        self.model_parameter = model_parameter

        self.timestepper = ImplicitEulerTimeStepper(
            nt = self.dims['nt']
        )

        assert self.model_parameter['T_final'] > self.model_parameter['T_initial']
        self.delta_t = (self.model_parameter['T_final'] - self.model_parameter['T_initial']) / self.dims['nt']

    def solve_state(self, q: VectorArray) -> VectorArray:
        assert q in self.Q
        assert isinstance(q, VectorArray)
        assert len(q) == (self.dims['nt'])

        iterator = self.timestepper.iterate(initial_time = self.model_parameter['T_initial'], 
                                            end_time = self.model_parameter['T_final'], 
                                            initial_data = self.u_0, 
                                            q=q,
                                            operator = self.A, 
                                            rhs=self.f, 
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

        for x in [q,u]:
            assert isinstance(x, VectorArray)
            assert len(x) == (self.dims['nt'])

        rhs = self.bilinear_cost_term.apply(u) - self.linear_cost_term.as_range_array()
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
        return self.V.make_array(np.flip(p.to_numpy()))
        

    def objective(self, 
                  u: Union[VectorArray, np.ndarray]) -> float:

        assert isinstance(u, (VectorArray, np.ndarray))
        assert len(u) == (self.dims['nt'])

        # Remove the vector at k = 0
        return  0.5 * self.delta_t * np.sum( \
                      self.bilinear_cost_term.pairwise_apply2(u,u) + \
                      (-2)  * self.linear_cost_term.as_range_array().pairwise_inner(u) + \
                      self.constant_cost_term
                    )

    def gradient(self,
                 u: VectorArray,
                 p: VectorArray) -> VectorArray:
        
        assert u in self.V
        assert p in self.V

        for x in [u,p]:
            assert isinstance(x, VectorArray)
            assert len(x) == (self.dims['nt'])

        grad = np.empty((self.dims['nt'], self.dims['state_dim']))
        
        # TODO Check if this is efficent and / or how its efficeny can be improved
        for idx in range(0, self.dims['nt']):
            grad[idx] = self.B(u[idx]).B_u_ad(p[idx], 'grad') 

        return self.Q.make_array(grad)


    def solve_linearized_state(self,
                               q: VectorArray,
                               d: VectorArray,
                               u: VectorArray) -> VectorArray:
        
        assert q in self.Q
        assert d in self.Q
        assert u in self.V

        for x in [q,d,u]:
            assert isinstance(x, VectorArray)
            assert len(x) == (self.dims['nt'])
        
        # TODO Check if this is efficent and / or how its efficeny can be improved
        rhs = self.V.make_array(np.array([
            -self.B(u[idx]).B_u(d[idx]).to_numpy()[0] for idx in range(len(d))
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

        rhs = self.bilinear_cost_term.apply(u + lin_u) - self.linear_cost_term.as_range_array()
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
        return self.V.make_array(np.flip(lin_p.to_numpy()))
    
    def linearized_objective(self,
                             q: VectorArray,
                             d: VectorArray,
                             u: VectorArray,
                             lin_u: VectorArray,
                             alpha : float) -> float:
    
        for x in [q, d,u,lin_u]:
            assert isinstance(x, VectorArray)
            assert len(x) == (self.dims['nt'])

        assert q in self.Q
        assert d in self.Q
        assert u in self.V
        assert lin_u in self.V

        #TODO Split into parts
        q_ = q + d - self.q_circ
        regularization_term = self.products['prod_Q'].pairwise_apply2(q_, q_)
        u_q_d = u + lin_u

        #TODO
        # -Check with Q-norm
        # Script for michael

        return  0.5 * self.delta_t * np.sum( \
                      self.bilinear_cost_term.pairwise_apply2(u_q_d,u_q_d) + \
                      (-2)  * self.linear_cost_term.as_range_array().pairwise_inner(u_q_d) + \
                      self.constant_cost_term
                      + alpha * regularization_term)


    def linearized_gradient(self,
                            q: VectorArray,
                            d: VectorArray,
                            u: VectorArray,
                            lin_p: VectorArray,
                            alpha : float) -> VectorArray:
        
        for x in [q, d, u, lin_p]:
            assert isinstance(x, VectorArray)
            assert len(x) == (self.dims['nt'])

        assert q in self.Q
        assert d in self.Q
        assert u in self.V
        assert lin_p in self.V

        grad = np.empty((self.dims['nt'], self.dims['state_dim']))
        
        # TODO Check if this is efficent and / or how its efficeny can be improved
        for idx in range(0, self.dims['nt']):
            grad[idx] = - self.B(u[idx]).B_u_ad(lin_p[idx], 'grad') 
        grad = self.Q.make_array(grad)

        return grad + alpha * self.products['prod_Q'].apply(q + d - self.q_circ)

    def linearized_hessian(self):
        raise NotImplementedError
                             
    def compute_objective(self, 
                          q: VectorArray) -> float:

        u = self.solve_state(q)
        return self.objective(u)
    
    def compute_gradient(self,
                         q: VectorArray) -> float:
        u = self.solve_state(q)
        p = self.solve_adjoint(q, u)
        return self.gradient(u, p)
        

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