from typing import Dict
from abc import ABC, abstractmethod
from enum import Enum

from pymor.vectorarrays.interface import VectorArray
from pymor.operators.interface import Operator 
from pymor.vectorarrays.interface import VectorSpace

from RBInvParam.error_estimators.residuals import AdjointResidualOperator
from RBInvParam.error_estimators.objective_error_estimators import CoercivityConstantEstimator

class AdjointErrorEstimatorType(Enum):
    NONE = "none"
    PARABOLIC = "parabolic"


class AdjointErrorEstimator(ABC):
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

        self.delta_t = self.setup['delta_t']
        self.nt = self.setup['dims']['nt']
        self.q_time_dep = self.setup['q_time_dep']

        self.A_coercivity_constant_estimator = A_coercivity_constant_estimator
        # assert self.Q == self.A_coercivity_constant_estimator.Q
        # assert self.Q == self.adjoint_residual_operator.Q
        # assert self.V == self.adjoint_residual_operator.V
        if product:
            assert self.adjoint_residual_operator.range == product.source
    
    @abstractmethod
    def compute_residuum(self, 
                         q: VectorArray,
                         u: VectorArray,
                         p: VectorArray,
                         use_cached_operators: bool = False,
                         cached_operators: Dict = None) -> VectorArray:
        pass

    @abstractmethod
    def estimate_error(self, 
                       q: VectorArray,
                       u: VectorArray,
                       p: VectorArray) -> VectorArray:
        pass

class ParabolicAdjointErrorEstimator(AdjointErrorEstimator):
    def __init__(self,
                 adjoint_residual_operator : AdjointResidualOperator,
                 A_coercivity_constant_estimator: CoercivityConstantEstimator,
                 Q : VectorSpace,
                 V : VectorSpace,
                 product : Operator,
                 setup: Dict):
    
        super().__init__(adjoint_residual_operator,
                         A_coercivity_constant_estimator,
                         Q, 
                         V, 
                         product, 
                         setup)

    def compute_residuum(self, 
                         q: VectorArray,
                         u: VectorArray,
                         p: VectorArray,
                         use_cached_operators: bool = False,
                         cached_operators: Dict = None) -> VectorArray:

        if self.q_time_dep:
            assert len(q) == (self.nt + 1)
        else:
            assert len(q) == 1

        assert q in self.Q
        assert u in self.V
        assert p in self.V
        assert len(u) == (self.nt + 1)
        assert len(p) == (self.nt + 1)

        p_old = self.V.empty()
        p_old.append(p[1:])
        p_old.append(self.V.zeros(count=1))

        return self.adjoint_residual_operator.apply(
            p = p, 
            p_old = p_old,
            u = u,
            q = q,
            use_cached_operators = use_cached_operators,
            cached_operators = cached_operators
        )
        
    
    def estimate_error(self, 
                       q: VectorArray,
                       u: VectorArray,
                       p: VectorArray) -> VectorArray:
        
        
        if self.q_time_dep:
            assert len(q) == (self.nt + 1)
        else:
            assert len(q) == 1

        assert q in self.Q
        assert u in self.V
        assert p in self.V
        assert len(u) == (self.nt + 1)
        assert len(u) == len(p)

        raise NotImplementedError

        
def create_adjoint_error_estimator(estimator_type: AdjointErrorEstimatorType,
                                   adjoint_residual_operator: AdjointResidualOperator,
                                   A_coercivity_constant_estimator: CoercivityConstantEstimator,
                                   Q: VectorSpace,
                                   V: VectorSpace,
                                   product: Operator,
                                   setup: Dict) -> AdjointErrorEstimator:
    """
    Factory function to create a StateErrorEstimator subclass
    based on the estimator_type enum.
    """
    if estimator_type == AdjointErrorEstimatorType.NONE:
        return None
    elif estimator_type == AdjointErrorEstimatorType.PARABOLIC:
        return ParabolicAdjointErrorEstimator(
            adjoint_residual_operator,
            A_coercivity_constant_estimator,
            Q, 
            V, 
            product, 
            setup
        )
    
    raise ValueError(f"Unsupported estimator type: {estimator_type}")