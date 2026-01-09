import copy
import numpy as np
from types import SimpleNamespace
from typing import List, Dict

import pymor_dealii_bindings as pd2


import pymor.vectorarrays as VectorArray

from pymor.vectorarrays.interface import VectorSpace
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.list import ListVectorArray
from pymor.vectorarrays.numpy import NumpyVectorArray


from RBInvParam.evaluators import FOMEvaluatorA, FOMEvaluatorB, B_u
from RBInvParam.problems.elasticity.elasticity_model import ElasticityModel
from RBInvParam.problems.shared.pymor_dealii_bindings.operator import DealIIBaseOperator, DealIIMatrixOperator
from RBInvParam.problems.shared.pymor_dealii_bindings.vectorarray import DealIIVectorSpace


class ElasticitiyFOMEvaluatorA(FOMEvaluatorA):
    def __init__(self,
                 elasticity_model: ElasticityModel,
                 source : VectorSpace,
                 range : VectorSpace,
                 Q : VectorSpace,
                 parameter_names: List[str] | None = None):
        assert isinstance(elasticity_model, ElasticityModel)
        self.elasticity_model = elasticity_model
        super().__init__(source, range, Q, parameter_names)
        
    
    def __call__(self, q: VectorArray, u: VectorArray = None) -> Dict:
        assert q in self.Q
        assert len(q) == 1

        self.elasticity_model.assemble_A_q(q.to_numpy().flatten())
        
        return {
            'A_q' : DealIIBaseOperator(op = self.elasticity_model.m_A_q),
            'partial_q_A_q_u' : None,
            #'partial_u_A_q_u' : DealIIBaseOperator(op = self.elasticity_model.m_partial_u_A_q_u),
            'partial_u_A_q_u' : None,
        }
    
    def clear_rhs_boundary_dofs(self, 
                                rhs: VectorArray,
                                flip: bool = False) -> VectorArray:
        
        assert isinstance(rhs, ListVectorArray)        
        for v in rhs.vectors:
            self.material_model.clear_rhs_boundary_dofs(v.real_part.impl)

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
        # if self.material_model.m_has_translation_operator:
        #     q = np.zeros((self.material_model.param_space_dim))
        #     self.material_model.m_q[:] = q
        #     self.material_model.assemble_system_matrix()
        #     self.system_matrix = self.material_model.system_matrix
        #     return DealIIMatrixOperator(self.system_matrix)
        # else:
        #     return None

    def get_parameteric_operator(self, q: VectorArray) -> Operator:
        raise NotImplementedError
        # self.material_model.m_q[:] = q.to_numpy()
        # self.material_model.assemble_parameteric_matrix()
        # self.system_matrix = self.material_model.system_matrix
        # return DealIIMatrixOperator(self.system_matrix)

class ElasticitiyFOMEvaluatorB(FOMEvaluatorB):
    def __init__(self,
                 source : VectorSpace,
                 range : VectorSpace,
                 Q : VectorSpace,
                 V : VectorSpace,
                 elasticity_model: ElasticityModel):
        
        super().__init__(source, range, Q, V)

        self._Q = DealIIVectorSpace(Q.dim)
        self.elasticity_model = elasticity_model

    def __call__(self, 
                 u: ListVectorArray,
                 parameter_basis_idx: int) -> B_u:
        assert u in self.V
        # TODO Check how this function can be vectorized
        assert len(u) == 1
        assert isinstance(u, ListVectorArray)
        
        #parameter_basis_idx = 0
        self.material_model.assemble_system_matrix_derivative(u.vectors[0].real_part.impl, parameter_basis_idx)
        B_u_op = DealIIMatrixOperator(
            matrix = self.material_model.system_matrix_derivatives[parameter_basis_idx]
        )

        def _B_u(d: NumpyVectorArray) -> pd2.Vector:
            # TODO Move parameter space handling to C++ and use pd2.Vector
            d_ = self._Q.from_numpy(d.to_numpy())
            return B_u_op.apply(d_).vectors[0].real_part.impl
            
        def _B_u_ad(p: ListVectorArray) -> np.ndarray:
            return B_u_op.apply_adjoint(p).to_numpy()

        return SimpleNamespace(B_u=_B_u, B_u_ad=_B_u_ad)