from typing import Dict
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

from pymor.vectorarrays.interface import VectorArray
from pymor.operators.interface import Operator 
from pymor.vectorarrays.interface import VectorSpace

from RBInvParam.error_estimators.residuals import StateResidualOperator
from RBInvParam.error_estimators.objective_error_estimators import CoercivityConstantEstimator

class StateErrorEstimatorType(Enum):
    NONE = "none"
    PARABOLIC = "parabolic"

class StateErrorEstimator(ABC):
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

        self.delta_t = self.setup['delta_t']
        self.nt = self.setup['dims']['nt']
        self.q_time_dep = self.setup['q_time_dep']
        
        self.A_coercivity_constant_estimator = A_coercivity_constant_estimator
        #assert self.Q == self.A_coercivity_constant_estimator.Q
        #assert self.Q == self.state_residual_operator.Q
        #assert self.V == self.state_residual_operator.V
        if product:
            assert self.state_residual_operator.range == product.source

    @abstractmethod
    def compute_residuum(self, 
                         q: VectorArray,
                         u: VectorArray,
                         use_cached_operators: bool = False,
                         cached_operators: Dict = None) -> VectorArray:
        pass

    @abstractmethod
    def compute_residuum(self, 
                         q: VectorArray,
                         u: VectorArray,
                         use_cached_operators: bool = False,
                         cached_operators: Dict = None) -> VectorArray:
        pass

class ParabolicStateErrorEstimator(StateErrorEstimator):
    def __init__(self,
                 state_residual_operator: StateResidualOperator,
                 A_coercivity_constant_estimator: CoercivityConstantEstimator,
                 Q: VectorSpace,
                 V: VectorSpace,
                 product: Operator,
                 setup: Dict) -> None:
    
        super().__init__(state_residual_operator,
                         A_coercivity_constant_estimator,
                         Q, 
                         V, 
                         product, 
                         setup)



    def compute_residuum(self, 
                         q: VectorArray,
                         u: VectorArray,
                         use_cached_operators: bool = False,
                         cached_operators: Dict = None) -> VectorArray:
        
        if self.q_time_dep:
            assert len(q) == (self.nt + 1)
        else:
            assert len(q) == 1

        assert q in self.Q
        assert u in self.V
        assert len(u) == (self.nt + 1)

        u_old = self.V.zeros(count=1)
        u_old.append(u[:-1])

        r = self.state_residual_operator.apply(
            u = u, 
            u_old = u_old,
            q = q,
            use_cached_operators = use_cached_operators,
            cached_operators = cached_operators
        )
        return r
        

    def estimate_error(self, 
                       q: VectorArray,
                       u: VectorArray,
                       use_cached_operators: bool = False,
                       cached_operators: Dict = None) -> float:
        
        if self.q_time_dep:
            assert len(q) == (self.nt + 1)
        else:
            assert len(q) == 1

        assert q in self.Q
        assert u in self.V
        assert len(u) == (self.nt + 1)

        alpha_q = np.min(self.A_coercivity_constant_estimator(q))
        r = self.compute_residuum(q=q, 
                                  u=u, 
                                  use_cached_operators = use_cached_operators,
                                  cached_operators = cached_operators)
        
        return np.sqrt(self.delta_t / alpha_q * np.sum(r.norm2(product=self.product)))
             
def create_state_error_estimator(estimator_type: StateErrorEstimatorType,
                                 state_residual_operator : StateResidualOperator,
                                   A_coercivity_constant_estimator: CoercivityConstantEstimator,
                                   Q: VectorSpace,
                                   V: VectorSpace,
                                   product: Operator,
                                   setup: Dict) -> StateErrorEstimator:
    """
    Factory function to create a StateErrorEstimator subclass
    based on the estimator_type enum.
    """
    if estimator_type == StateErrorEstimatorType.NONE:
        return None
    elif estimator_type == StateErrorEstimatorType.PARABOLIC:
        return ParabolicStateErrorEstimator(
            state_residual_operator,
            A_coercivity_constant_estimator,
            Q, 
            V, 
            product, 
            setup
        )
    
    raise ValueError(f"Unsupported estimator type: {estimator_type}")