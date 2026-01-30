from typing import List

import pymor.vectorarrays as VectorArray

from pymor.vectorarrays.interface import VectorSpace
from pymor.operators.interface import Operator
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
                 Q : VectorSpace,
                 parameter_names: List[str] | None = None):
        
        assert isinstance(hyperelasticity_model, HyperElasticityModel)
        self.hyperelasticity_model = hyperelasticity_model
        super().__init__(source, range, Q, parameter_names)
        

    def get_A_q(self, q: VectorArray) -> Operator:
        assert q in self.Q
        assert len(q) == 1

        op = self.hyperelasticity_model.assemble_A_q(
            q_np = q.to_numpy().flatten()
        )
        # TODO Remove linear from FOMEvaluatorA
        self.linear = op.linear
        
        return DealIIBaseOperator(
            op = op
        )
        
    def get_partial_q_A_q_u(self, q: VectorArray , u: VectorArray) -> Operator:
        assert q in self.Q
        assert len(q) == 1

        assert u in self.source
        assert len(u) == 1

        raise NotImplementedError
        
    def get_partial_u_A_q_u(self, q: VectorArray , u: VectorArray, A_q: Operator = None) -> Operator:
        assert q in self.Q
        assert len(q) == 1
        assert u in self.source
        assert len(u) == 1

        if A_q is None:        
            A_q = self.hyperelasticity_model.assemble_A_q(
                q_np = q.to_numpy().flatten()
            )
        
        assert isinstance(A_q, Operator)
        
        try:
            return A_q.jacobian(U=u)
        except:
            raise InvalidAssemblyArgument
    
    def clear_rhs_boundary_dofs(self, 
                                rhs: VectorArray,
                                flip: bool = False) -> VectorArray:
        
        assert isinstance(rhs, ListVectorArray)        
        for v in rhs.vectors:
            self.hyperelasticity_model.clear_rhs_boundary_dofs(v.impl)

        if flip:
            return self.flip_vector_array(rhs)
        else:
            return rhs

    def flip_vector_array(self, vector_array: VectorArray) -> VectorArray:
        assert isinstance(vector_array, ListVectorArray)
        vector_array = vector_array.space.make_array(vector_array.vectors[::-1])
        return vector_array

    def get_translation_operator(self) -> Operator | None:
        raise NotImplementedError
    
    def get_parameteric_operator(self, q: VectorArray) -> Operator:
        assert q in self.Q
        assert len(q) == 1
        
        raise NotImplementedError
   