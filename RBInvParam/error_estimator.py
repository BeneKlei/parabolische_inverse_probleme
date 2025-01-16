from abc import abstractmethod
from typing import Dict
import numpy as np

from pymor.vectorarrays.interface import VectorArray
from pymor.parameters.functionals import MinThetaParameterFunctional
from pymor.operators.interface import Operator 
from pymor.vectorarrays.interface import VectorSpace

from RBInvParam.residuals import StateResidualOperator, AdjointResidualOperator
    
class StateErrorEstimator():
    def __init__(self,
                 state_residual_operator : StateResidualOperator,
                 coercivity_estimator : MinThetaParameterFunctional,
                 Q : VectorSpace,
                 V : VectorSpace,
                 product : Operator,
                 setup: Dict):
        
        assert isinstance(state_residual_operator, StateResidualOperator)
        #assert isinstance(coercivity_estimator, MinThetaParameterFunctional)
        
        self.state_residual_operator = state_residual_operator
        self.coercivity_estimator = coercivity_estimator
        self.Q = Q
        self.V = V
        self.product = product
        self.setup = setup

        self.delta_t = self.setup['model_parameter']['delta_t']
        self.nt = self.setup['dims']['nt']
        self.q_time_dep = self.setup['model_parameter']['q_time_dep']

        assert self.Q == self.state_residual_operator.Q
        assert self.V == self.state_residual_operator.V
        if product:
            assert self.state_residual_operator.range == product.source

    
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

        # TODO THIS IS NOT CORRECT YET.
        alpha_q = 1
        # alpha_q = self.coercivity_estimator.estimate(...)
        u_old = self.V.zeros(count=1)
        u_old.append(u[:-1])

        # TODO THIS IS NOT CORRECT YET. Product is wrong
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

        # TODO THIS IS NOT CORRECT YET.
        alpha_q = 1
        # alpha_q = self.coercivity_estimator.estimate(...)

        p_old = self.V.empty()
        p_old.append(p[1:])
        p_old.append(self.V.zeros(count=1))

        # print(p_old.to_numpy()[:,20])
        # print(p.to_numpy()[:,20])

        # TODO THIS IS NOT CORRECT YET. Product is wrong
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
