from typing import Dict, Callable
import numpy as np

from pymor.vectorarrays.interface import VectorArray
from pymor.operators.interface import Operator 
from pymor.vectorarrays.interface import VectorSpace

from RBInvParam.residuals import StateResidualOperator, AdjointResidualOperator

class CoercivityEstimator():
    def __init__(self, 
                 coercivity_estimator_function: Callable,
                 Q: VectorSpace):
        
        self.coercivity_estimator_function = coercivity_estimator_function
        self.Q = Q

    def __call__(self, q: VectorArray) -> float:
        assert q in self.Q
        ret = self.coercivity_estimator_function(q)
        assert isinstance(ret, float)
        assert ret > 0
        return ret
    
    
class StateErrorEstimator():
    def __init__(self,
                 state_residual_operator : StateResidualOperator,
                 Q : VectorSpace,
                 V : VectorSpace,
                 product : Operator,
                 setup: Dict):
        
        assert isinstance(state_residual_operator, StateResidualOperator)
        
        self.state_residual_operator = state_residual_operator
        self.Q = Q
        self.V = V
        self.product = product
        self.setup = setup

        self.delta_t = self.setup['model_parameter']['delta_t']
        self.nt = self.setup['dims']['nt']
        self.q_time_dep = self.setup['model_parameter']['q_time_dep']

        self.setup_coercivity_estimator()

        assert self.Q == self.state_residual_operator.Q
        assert self.V == self.state_residual_operator.V
        if product:
            assert self.state_residual_operator.range == product.source

    def setup_coercivity_estimator(self) -> None:
        pass
        #raise NotImplementedError
        # problem_type = self.setup['problem_parameter']['problem_type']
        # assert self.setup['problem_parameter']['state_space_product'] == 'h1'
        
        # if 'dirichlet' in problem_type and 'diffusion' in problem_type:
        #     coercivity_estimator_function = lambda q: abs(min(q.to_numpy()[0]))
        # elif 'dirichlet' in problem_type and 'reaction' in problem_type:
        #     coercivity_estimator_function = lambda q: 1
        # else:
        #     raise ValueError('No matching problemtype given')
        
        # self.coercivity_estimator = CoercivityEstimator(
        #          coercivity_estimator_function = coercivity_estimator_function,
        #          Q = self.Q)
    
    def estimate_error(self, 
                       q: VectorArray,
                       u: VectorArray) -> float:
        
        if self.q_time_dep:
            assert len(q) == self.nt
        else:
            assert len(q) == 1

        assert q in self.Q
        assert u in self.V
        assert len(u) == self.nt

        if self.q_time_dep:
            alpha_qks = np.array([
                self.coercivity_estimator(qk) for qk in q
            ])
            alpha_q = np.min(alpha_qks)
        else:
            alpha_q = self.coercivity_estimator(q[0])
            
        u_old = self.V.zeros(count=1)
        u_old.append(u[:-1])

        r = self.state_residual_operator.apply(
            u = u, 
            u_old = u_old,
            q = q
        ).norm2(product=self.product)

        return np.sqrt(self.delta_t / alpha_q * np.sum(r))
             
class AdjointErrorEstimator():
    def __init__(self,
                 adjoint_residual_operator : AdjointResidualOperator,
                 Q : VectorSpace,
                 V : VectorSpace,
                 product : Operator,
                 setup: Dict):
    
        assert isinstance(adjoint_residual_operator, AdjointResidualOperator)
        
        self.adjoint_residual_operator = adjoint_residual_operator
        self.Q = Q
        self.V = V
        self.product = product
        self.setup = setup

        self.delta_t = self.setup['model_parameter']['delta_t']
        self.nt = self.setup['dims']['nt']
        self.q_time_dep = self.setup['model_parameter']['q_time_dep']

        assert self.Q == self.adjoint_residual_operator.Q
        assert self.V == self.adjoint_residual_operator.V
        if product:
            assert self.adjoint_residual_operator.range == product.source
        
        
    def estimate_error(self, 
                       q: VectorArray,
                       u: VectorArray,
                       p: VectorArray) -> float:
        
        if self.q_time_dep:
            assert len(q) == self.nt
        else:
            assert len(q) == 1

        assert q in self.Q
        assert u in self.V
        assert p in self.V
        assert len(u) == self.nt
        assert len(u) == len(p)

        if self.q_time_dep:
            alpha_qks = np.array([
                self.coercivity_estimator(qk) for qk in q
            ])
            alpha_q = np.min(alpha_qks)
        else:
            alpha_q = self.coercivity_estimator(q[0])
        
        p_old = self.V.empty()
        p_old.append(p[1:])
        p_old.append(self.V.zeros(count=1))

        r = self.adjoint_residual_operator.apply(
            p = p, 
            p_old = p_old,
            u = u,
            q = q
        ).norm2(product=self.product)

        return np.sqrt(self.delta_t / alpha_q * np.sum(r))

# class ObjectiveErrorEstimator():
#     def __init__(self):
#         pass
