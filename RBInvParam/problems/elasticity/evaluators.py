import copy
import numpy as np
from types import SimpleNamespace
from typing import List

import pymor_dealii_bindings as pd2


import pymor.vectorarrays as VectorArray

from pymor.vectorarrays.interface import VectorSpace
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.list import ListVectorArray
from pymor.vectorarrays.numpy import NumpyVectorArray


from RBInvParam.evaluators import FOMEvaluatorA, FOMEvaluatorB, BU
from RBInvParam.problems.elasticity.material_model import MaterialModel
from RBInvParam.problems.elasticity.pymor_dealii_bindings.operator import DealIIMatrixOperator
from RBInvParam.problems.elasticity.pymor_dealii_bindings.vectorarray import DealIIVectorSpace


class ElasticitiyFOMEvaluatorA(FOMEvaluatorA):
    def __init__(self,
                 material_model: MaterialModel,
                 source : VectorSpace,
                 range : VectorSpace,
                 Q : VectorSpace,
                 parameter_names: List[str] | None):
        
        super().__init__(source, range, Q, parameter_names)

        self.material_model = material_model
        self.system_matrix = None
        #self.sparsity_pattern = self.material_model.system_matrix_sp
    
    def __call__(self, q: VectorArray) -> Operator:
        assert q in self.Q
        
        #self.system_matrix.reinit(self.sparsity_pattern)
        self.material_model.m_q[:] = q.to_numpy()
        self.material_model.assemble_system_matrix()
        self.system_matrix = self.material_model.system_matrix
        return DealIIMatrixOperator(self.system_matrix)
            
    
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
        if self.material_model.m_has_translation_operator:
            q = np.zeros((self.material_model.param_space_dim))
            self.material_model.m_q[:] = q
            self.material_model.assemble_system_matrix()
            self.system_matrix = self.material_model.system_matrix
            return DealIIMatrixOperator(self.system_matrix)
        else:
            return None

    def get_parameteric_operator(self, q: VectorArray) -> Operator:



class ElasticitiyFOMEvaluatorB(FOMEvaluatorB):
    def __init__(self,
                 source : VectorSpace,
                 range : VectorSpace,
                 Q : VectorSpace,
                 V : VectorSpace,
                 material_model: MaterialModel):
        
        super().__init__(source, range, Q, V)

        self.Q_ = DealIIVectorSpace(Q.dim)
        self.material_model = material_model

    def __call__(self, 
                 u: ListVectorArray,
                 parameter_basis_idx: int) -> BU:
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
            d_ = self.Q_.from_numpy(d.to_numpy())
            return B_u_op.apply(d_).vectors[0].real_part.impl
            
        def _B_u_ad(p: ListVectorArray) -> np.ndarray:
            return B_u_op.apply_adjoint(p).to_numpy()

        return SimpleNamespace(B_u=_B_u, B_u_ad=_B_u_ad)