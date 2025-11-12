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
    HYPERBOLIC = "hyperbolic"

class StateErrorEstimator():
    def __init__(self,
                 state_residual_operator : StateResidualOperator,
                 A_coercivity_constant_estimator: CoercivityConstantEstimator,
                 Q : VectorSpace,
                 V : VectorSpace,
                 gram_operator : Operator,
                 setup: Dict):
        
        assert isinstance(state_residual_operator, StateResidualOperator)
        
        self.state_residual_operator = state_residual_operator
        self.Q = Q
        self.V = V
        self.gram_operator = gram_operator
        self.setup = setup

        self.delta_t = self.setup['delta_t']
        self.nt = self.setup['dims']['nt']
        self.q_time_dep = self.setup['q_time_dep']
        
        self.A_coercivity_constant_estimator = A_coercivity_constant_estimator
        #assert self.Q == self.A_coercivity_constant_estimator.Q
        #assert self.Q == self.state_residual_operator.Q
        #assert self.V == self.state_residual_operator.V
        if gram_operator:
            assert self.state_residual_operator.range == gram_operator.source

    @abstractmethod
    def estimate_error(self, 
                    q: VectorArray,
                    u: VectorArray,
                    u_dot: VectorArray = None,
                    use_cached_operators: bool = False,
                    cached_operators: Dict = None) -> float:
        pass

class ParabolicStateErrorEstimator(StateErrorEstimator):
    def estimate_error(self, 
                       q: VectorArray,
                       u: VectorArray,
                       u_dot: VectorArray = None,
                       use_cached_operators: bool = False,
                       cached_operators: Dict = None) -> float:
        
        if self.q_time_dep:
            assert len(q) == (self.nt + 1)
        else:
            assert len(q) == 1

        assert q in self.Q
        assert u in self.V
        assert len(u) == (self.nt + 1)

        r = self.state_residual_operator.apply(
            u = u, 
            mass_u = u,
            q = q,
            use_cached_operators = use_cached_operators,
            cached_operators = cached_operators
        )
        
        alpha_q = np.min(self.A_coercivity_constant_estimator(q))
        return np.sqrt(self.delta_t / alpha_q * np.sum(r.norm2(product=self.gram_operator)))
             

class HyperbolicStateErrorEstimator(StateErrorEstimator):
    def estimate_error(self, 
                       q: VectorArray,
                       u: VectorArray,
                       u_dot: VectorArray,
                       use_cached_operators: bool = False,
                       cached_operators: Dict = None) -> float:
        
        if self.q_time_dep:
            assert len(q) == (self.nt + 1)
        else:
            assert len(q) == 1

        assert q in self.Q
        assert u in self.V
        assert u_dot in self.V

        assert len(u) == (self.nt + 1)
        assert len(u_dot) == (self.nt + 1)

        r = self.state_residual_operator.apply(
            u = u, 
            mass_u = u_dot,
            q = q,
            use_cached_operators = use_cached_operators,
            cached_operators = cached_operators
        )
        r = self.gram_operator.pairwise_apply2(r,r)
        assert r.shape == (self.nt,)
        r = np.sqrt(r)
        inner_sums = np.cumsum(r)
        inner_sums *= self.delta_t
        inner_sums = inner_sums**2
        err = np.sum(inner_sums)
        err = np.sqrt(err)
        err *= 2 * np.sqrt(self.delta_t)
        return err


def create_state_error_estimator(estimator_type: StateErrorEstimatorType,
                                 products : Dict,
                                 **kwargs) -> StateErrorEstimator:
    """
    Factory function to create a StateErrorEstimator subclass
    based on the estimator_type enum.
    """
    if estimator_type == StateErrorEstimatorType.NONE:
        return None
    elif estimator_type == StateErrorEstimatorType.PARABOLIC:
        return ParabolicStateErrorEstimator(**kwargs, gram_operator=products['prod_V'])
    elif estimator_type == StateErrorEstimatorType.HYPERBOLIC:
        return HyperbolicStateErrorEstimator(**kwargs, gram_operator=products['prod_H'])
    
    raise ValueError(f"Unsupported estimator type: {estimator_type}")

