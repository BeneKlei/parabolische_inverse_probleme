from typing import Dict, Union
from numbers import Number
import numpy as np

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.interface import VectorArray
from pymor.operators.interface import Operator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from evaluators import A_evaluator, B_evaluator
from timestepping import ImplicitEulerTimeStepper

# TODO 
# - Enbale auto conversion to NumpyVectorArray / Numpyvector
# - And assert correct space
# - Add caching


# Notes caching:
# - Is q t dep A(q) can be stored for all time steps or calculated on the fly.
#   - The later is maybe possible for ROMs not for FOM
# 

class InstationaryModelIP:
    def __init__(
        self,
        u_0 : VectorArray, 
        M : Operator,
        A : A_evaluator,
        f : VectorArray,
        B : B_evaluator,
        constant_cost_term: Number,
        linear_cost_term: NumpyMatrixOperator,
        bilinear_cost_term: NumpyMatrixOperator,
        Q_h : NumpyVectorSpace,
        q_circ: VectorArray,
        constant_reg_term: Number,
        linear_reg_term: NumpyMatrixOperator,
        bilinear_reg_term: NumpyMatrixOperator,
        products : Dict,
        visualizer,
        dims: Dict,
        model_parameter: Dict,
    ):
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

        self.V_h = M.source
        self.Q_h = Q_h

        
    def solve_state(self, q: Union[VectorArray, np.ndarray]) -> VectorArray:
        assert isinstance(q, (VectorArray, np.ndarray))
        assert len(q) == (self.dims['nt'] + 1)

        iterator = self.timestepper.iterate(initial_time = self.model_parameter['T_initial'], 
                                            end_time = self.model_parameter['T_final'], 
                                            initial_data = self.u_0, 
                                            q=q,
                                            operator = self.A, 
                                            rhs=self.f, 
                                            mass=self.M)
        
        u = self.V_h.empty(reserve= self.dims['nt'] + 1)
        for u_n, _ in iterator:
            u.append(u_n)
        return u

    def solve_adjoint(self, 
                      q: Union[VectorArray, np.ndarray], 
                      u: Union[VectorArray, np.ndarray]) -> VectorArray:
        
        assert isinstance(q, (VectorArray, np.ndarray))
        assert isinstance(u, (VectorArray, np.ndarray))
        assert len(u) == (self.dims['nt'] + 1)

        rhs = self.bilinear_cost_term.apply(u) - self.linear_cost_term.as_range_array()
        iterator = self.timestepper.iterate(initial_time = self.model_parameter['T_initial'], 
                                            end_time = self.model_parameter['T_final'], 
                                            initial_data = self.p_0, 
                                            q=q,
                                            operator = self.A, 
                                            rhs=rhs, 
                                            mass=self.M,
                                            )
        
        p = self.V_h.empty(reserve= self.dims['nt'] + 1)
        for p_n, _ in iterator:
            p.append(p_n)
        return self.V_h.make_array(np.flip(p.to_numpy()))
        

    def objective(self, 
                  u: Union[VectorArray, np.ndarray]) -> Number:

        assert isinstance(u, (VectorArray, np.ndarray))
        assert len(u) == (self.dims['nt'] + 1)

        # Remove the vector at k = 0
        return  0.5 * self.delta_t * np.sum( \
                      self.bilinear_cost_term.pairwise_apply2(u[1:],u[1:]) + \
                      (-2)  * self.linear_cost_term.as_range_array().pairwise_inner(u)[1:] + \
                      self.constant_cost_term[1:]
                    )

    def gradient(self,
                 u: Union[VectorArray, np.ndarray],
                 p: Union[VectorArray, np.ndarray]) -> VectorArray:

        for x in [u,p]:
            assert isinstance(x, (VectorArray, np.ndarray))
            assert len(x) == (self.dims['nt'] + 1)

        assert u.space == self.V_h
        assert p.space == self.V_h

        grad = np.empty((self.dims['nt'], self.dims['state_dim']))
        
        # TODO Check if this is efficent and / or how its efficeny can be improved
        for idx in range(1, self.dims['nt'] + 1):
            grad[idx-1] = self.B(u[idx]).B_u_ad(p[idx], 'grad') 

        return self.Q_h.make_array(grad)


    def solve_linearized_state(self,
                               q: Union[VectorArray, np.ndarray],
                               d: Union[VectorArray, np.ndarray],
                               u: Union[VectorArray, np.ndarray]) -> VectorArray:

        for x in [q,d,u]:
            assert isinstance(x, (VectorArray, np.ndarray))
            assert len(x) == (self.dims['nt'] + 1)
        
        # TODO Check if this is efficent and / or how its efficeny can be improved
        rhs = self.V_h.make_array(np.array([
            -self.B(u).B_u(d_).to_numpy()[0] for d_ in d
        ]))
        

        iterator = self.timestepper.iterate(initial_time = self.model_parameter['T_initial'], 
                                            end_time = self.model_parameter['T_final'], 
                                            initial_data = self.linearized_u_0, 
                                            q=q,
                                            operator = self.A, 
                                            rhs=rhs, 
                                            mass=self.M)        
        lin_u = self.V_h.empty(reserve= self.dims['nt'] + 1)
        for lin_u_n, _ in iterator:
            lin_u.append(lin_u_n)
        return lin_u

    def solve_linearized_adjoint(self,
                                 q: Union[VectorArray, np.ndarray],
                                 u: Union[VectorArray, np.ndarray],
                                 lin_u: Union[VectorArray, np.ndarray]) -> VectorArray:

        
        rhs = self.bilinear_cost_term.apply(u + lin_u) - self.linear_cost_term.as_range_array()
        iterator = self.timestepper.iterate(initial_time = self.model_parameter['T_initial'], 
                                            end_time = self.model_parameter['T_final'], 
                                            initial_data = self.p_0, 
                                            q=q,
                                            operator = self.A, 
                                            rhs=rhs, 
                                            mass=self.M,
                                            )
        
        lin_p = self.V_h.empty(reserve= self.dims['nt'] + 1)
        for lin_p_n, _ in iterator:
            lin_p.append(lin_p_n)
        return self.V_h.make_array(np.flip(lin_p.to_numpy()))
    
    def linearized_objective(self,
                             q: Union[VectorArray, np.ndarray],
                             d: Union[VectorArray, np.ndarray],
                             u: Union[VectorArray, np.ndarray],
                             lin_u: Union[VectorArray, np.ndarray],
                             alpha : Number
                             ) -> Number:
    
        for x in [q, d,u,lin_u]:
            assert isinstance(x, (VectorArray, np.ndarray))
            assert len(x) == (self.dims['nt'] + 1)

        if not isinstance(q, VectorArray):
            q = self.Q_h.make_array(q)
        assert q in self.Q_h

        if not isinstance(d, VectorArray):
            d = self.Q_h.make_array(d)
        assert d in self.Q_h


        q_ = q + d + self.q_circ
        regularization_term = self.products['prod_Q'].pairwise_apply2(q_[1:], q_[1:])
        u_q_d = u + lin_u

        # Remove the vector at k = 0
        return  0.5 * self.delta_t * np.sum( \
                      self.bilinear_cost_term.pairwise_apply2(u_q_d[1:],u_q_d[1:]) + \
                      (-2)  * self.linear_cost_term.as_range_array().pairwise_inner(u_q_d)[1:] + \
                      self.constant_cost_term[1:] 
                      + alpha * regularization_term)


    def linearized_gradient(self,
                            q: Union[VectorArray, np.ndarray],
                            d: Union[VectorArray, np.ndarray],
                            u: VectorArray,
                            lin_p: VectorArray,
                            alpha : Number
                            ) -> VectorArray:
        
        raise NotImplementedError
        # if not isinstance(q, VectorArray):
        #     assert isinstance(q, np.ndarray)
        #     q = self.Q_h.make_array(q)

        # if not isinstance(d, VectorArray):
        #     assert isinstance(d, np.ndarray)
        #     d = self.Q_h.make_array(d)            
    
        # for x in [q, d, u, lin_u]:
        #     assert isinstance(x, VectorArray)
        #     assert len(x) == (self.dims['nt'] + 1)

        # assert q in self.Q_h
        # assert d in self.Q_h
        # assert u in self.V_h
        # assert lin_p in self.V_h

        # grad = np.empty((self.dims['nt'], self.dims['state_dim']))
        
        # # TODO Check if this is efficent and / or how its efficeny can be improved
        # for idx in range(1, self.dims['nt'] + 1):
        #     grad[idx-1] = self.B(u[idx]).B_u_ad(p[idx], 'grad') 
        # grad = self.Q_h.make_array(grad)

        #return grad + alpha * 

    def linearized_hessian(self):
        raise NotImplementedError