from typing import Dict, Callable
import numpy as np

from pymor.vectorarrays.interface import VectorArray
from pymor.operators.interface import Operator 
from pymor.vectorarrays.interface import VectorSpace

from RBInvParam.residuals import StateResidualOperator, AdjointResidualOperator

# TODO 
# - Write tests for error est.

class CoercivityConstantEstimator():
    def __init__(self, 
                 coercivity_estimator_function: Callable,
                 Q: VectorSpace,
                 q_time_dep: bool):
        
        self.coercivity_estimator_function = coercivity_estimator_function
        self.Q = Q
        self.q_time_dep = q_time_dep

    def __call__(self, q: VectorArray) -> np.ndarray:
        assert q in self.Q

        if self.q_time_dep:
            alpha_qks = np.array([
                self.coercivity_estimator_function(qk) for qk in q
            ])
        else:
            alpha_qks = np.array([self.coercivity_estimator_function(q[0])])
        
        assert np.all(alpha_qks > 0)
        return alpha_qks

    
class StateErrorEstimator():
    def __init__(self,
                 state_residual_operator : StateResidualOperator,
                 A_coercivity_constant_estimator: CoercivityConstantEstimator,
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
        
        self.A_coercivity_constant_estimator = A_coercivity_constant_estimator
        assert self.Q == self.A_coercivity_constant_estimator.Q
        assert self.Q == self.state_residual_operator.Q
        assert self.V == self.state_residual_operator.V
        if product:
            assert self.state_residual_operator.range == product.source
    
    def compute_residuum(self, 
                         q: VectorArray,
                         u: VectorArray) -> VectorArray:
        
        if self.q_time_dep:
            assert len(q) == self.nt
        else:
            assert len(q) == 1

        assert q in self.Q
        assert u in self.V
        assert len(u) == self.nt

        u_old = self.V.zeros(count=1)
        u_old.append(u[:-1])

        r = self.state_residual_operator.apply(
            u = u, 
            u_old = u_old,
            q = q
        )
        return r
        

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

        alpha_q = np.min(self.A_coercivity_constant_estimator(q))
        r = self.compute_residuum(q, u)
        return np.sqrt(self.delta_t / alpha_q * np.sum(r.norm2(product=self.product)))
             
class AdjointErrorEstimator():
    def __init__(self,
                 adjoint_residual_operator : AdjointResidualOperator,
                 A_coercivity_constant_estimator: CoercivityConstantEstimator,
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

        self.A_coercivity_constant_estimator = A_coercivity_constant_estimator
        assert self.Q == self.A_coercivity_constant_estimator.Q
        assert self.Q == self.adjoint_residual_operator.Q
        assert self.V == self.adjoint_residual_operator.V
        if product:
            assert self.adjoint_residual_operator.range == product.source
    
    def compute_residuum(self, 
                         q: VectorArray,
                         u: VectorArray,
                         p: VectorArray) -> VectorArray:

        p_old = self.V.empty()
        p_old.append(p[1:])
        p_old.append(self.V.zeros(count=1))

        return self.adjoint_residual_operator.apply(
            p = p, 
            p_old = p_old,
            u = u,
            q = q
        )
        
    
    def estimate_error(self, 
                       q: VectorArray,
                       u: VectorArray,
                       p: VectorArray) -> VectorArray:
        
        
        if self.q_time_dep:
            assert len(q) == self.nt
        else:
            assert len(q) == 1

        assert q in self.Q
        assert u in self.V
        assert p in self.V
        assert len(u) == self.nt
        assert len(u) == len(p)

        raise NotImplementedError

class ObjectiveErrorEstimator():
    def __init__(self,
                 A_coercivity_constant_estimator: CoercivityConstantEstimator,
                 C_continuity_constant: float):

        self.A_coercivity_constant_estimator = A_coercivity_constant_estimator
        self.C_continuity_constant = C_continuity_constant


    def estimate_error(self, 
                       q: VectorArray,
                       estimated_state_error: float,
                       adjoint_residuum: float) -> float:

        # TODO Is this correct? I do not think so 
        alpha_q = np.min(self.A_coercivity_constant_estimator(q))

        ret = 0
        ret += adjoint_residuum * estimated_state_error / np.sqrt(alpha_q)
        ret += self.C_continuity_constant**2 / (2 * alpha_q) * estimated_state_error**2

        return ret 
        
