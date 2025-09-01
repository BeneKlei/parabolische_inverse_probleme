from typing import Callable
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.interface import VectorSpace

class ObjectiveErrorEstimatorType(Enum):
    NONE = "none"
    PARABOLIC = "parabolic"

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


class ObjectiveErrorEstimator(ABC):
    def __init__(self,
                 A_coercivity_constant_estimator: CoercivityConstantEstimator,
                 C_continuity_constant: float):

        self.A_coercivity_constant_estimator = A_coercivity_constant_estimator
        self.C_continuity_constant = C_continuity_constant

    @abstractmethod
    def estimate_error(self, 
                    q: VectorArray,
                    estimated_state_error: float,
                    adjoint_residuum: float) -> float:
        pass
            
class ParabolicObjectiveErrorEstimator(ObjectiveErrorEstimator):
    def __init__(self,
                 A_coercivity_constant_estimator: CoercivityConstantEstimator,
                 C_continuity_constant: float):

        super().__init__(A_coercivity_constant_estimator,
                         C_continuity_constant)

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
        
def create_objective_error_estimator(estimator_type: ObjectiveErrorEstimatorType,
                                     A_coercivity_constant_estimator: CoercivityConstantEstimator,
                                     C_continuity_constant: float) -> ObjectiveErrorEstimator:
    """
    Factory function to create a StateErrorEstimator subclass
    based on the estimator_type enum.
    """
    if estimator_type == ObjectiveErrorEstimatorType.NONE:
        return None
    elif estimator_type == ObjectiveErrorEstimatorType.PARABOLIC:
        return ParabolicObjectiveErrorEstimator(
            A_coercivity_constant_estimator,
            C_continuity_constant
        )
    
    raise ValueError(f"Unsupported estimator type: {estimator_type}")