from typing import List

import pymor.vectorarrays as VectorArray

from pymor.vectorarrays.interface import VectorSpace
from pymor.operators.interface import Operator
from pymor.operators.constructions import ZeroOperator
from pymor.vectorarrays.list import ListVectorArray

from RBInvParam.evaluators import FOMEvaluatorA, InvalidAssemblyArgument
from RBInvParam.problems.hyperelasticity.hyperelasticity_model import HyperElasticityModel

#from hyperelasticity_model import HyperElasticityModel


from RBInvParam.problems.shared.pymor_dealii_bindings.operator import *


class HyperElasticitiyFOMEvaluatorA(FOMEvaluatorA):
    def __init__(self,
                 hyperelasticity_model: HyperElasticityModel,
                 source : VectorSpace,
                 range : VectorSpace,
                 Q : VectorSpace):
        
        assert isinstance(hyperelasticity_model, HyperElasticityModel)
        self.hyperelasticity_model = hyperelasticity_model
        super().__init__(
            source = source, 
            range = range, 
            Q = Q, 
            A_affine = self.hyperelasticity_model.A_affine(), 
            A_q_linear = self.hyperelasticity_model.A_q_linear())

        self._OperatorCls = SparseMatrixOperator if self.A_q_linear else DealIIBaseOperator    

    def get_A_q(self, q: VectorArray) -> Operator:
        assert q in self.Q
        assert len(q) == 1
    
        return self._OperatorCls(
            op=self.hyperelasticity_model.assemble_A_q(
                q_np=q.to_numpy().ravel()
            )
        )
        
    def get_partial_u_A_q_u(self, q: VectorArray , u: VectorArray, A_q: Operator = None) -> Operator:
        assert q in self.Q
        assert len(q) == 1
        if u is not None:
            assert u in self.source
            assert len(u) == 1

        if A_q is None:
             A_q = self._OperatorCls(
                op = self.hyperelasticity_model.assemble_A_q(
                    q_np = q.to_numpy().ravel()
                )
            )
            
        assert isinstance(A_q, Operator)
        
        try:
            return A_q.jacobian(U=u)
        except:
            raise InvalidAssemblyArgument
    
    def get_partial_q_A_q_u(self, q: VectorArray , u: VectorArray) -> Operator:
        assert q in self.Q
        assert len(q) == 1
        assert u in self.source
        assert len(u) == 1

        return DealIIBaseOperator(
            op = self.hyperelasticity_model.assemble_partial_q_A_q_u(
                q_np = q.to_numpy().ravel(),
                u = u.vectors[0].impl
            ),
            source_space="numpy",
            range_space="dealii"      
        )
    
    def clear_rhs_boundary_dofs(self, 
                                rhs: VectorArray,
                                flip: bool = False) -> VectorArray:
        
        assert isinstance(rhs, ListVectorArray)        
        for v in rhs.vectors:
            self.hyperelasticity_model.clear_rhs_boundary_dofs(v.impl)

        if flip:
            return rhs[::-1]
        else:
            return rhs

    def get_translation_operator(self) -> Operator:
        if self.A_affine:            
            return self.get_A_q(self.Q.zeros())
        else:
            return ZeroOperator(source=self.source, range=self.range)

    # TODO Rename this
    def get_parameteric_operator(self, q: VectorArray) -> Operator:
        assert q in self.Q
        assert len(q) == 1
    
        return self._OperatorCls(
            op = self.hyperelasticity_model.assemble_A_q(
                q_np = q.to_numpy().ravel(),
                param_linear_part_only = True
            )
        )
   