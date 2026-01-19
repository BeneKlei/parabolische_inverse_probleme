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


from RBInvParam.evaluators import FOMEvaluatorA
#, FOMEvaluatorB, B_u
from RBInvParam.problems.elasticity.elasticity_model import ElasticityModel
from RBInvParam.problems.shared.pymor_dealii_bindings.operator import *
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
        self.A_q_linear_op = True
        super().__init__(source, range, Q, parameter_names)
        

    def get_A_q(self, q: VectorArray) -> Operator:
        assert q in self.Q
        assert len(q) == 1
        
        return SparseMatrixOperator(
            op = self.elasticity_model.assemble_A_q(
                q_np = q.to_numpy().flatten()
            )
        )

    def get_partial_q_A_q_u(self, q: VectorArray , u: VectorArray) -> Operator:
        assert q in self.Q
        assert len(q) == 1

        assert u in self.source
        assert len(u) == 1

        return NumpyDealIIFullMatrixOperator(
            op = self.elasticity_model.assemble_partial_q_A_q_u(
                u = u.vectors[0].impl
            )
        )
        
    def get_partial_u_A_q_u(self, q: VectorArray , u: VectorArray) -> Operator:
        assert q in self.Q
        assert len(q) == 1
        
        assert u is None
        
        return SparseMatrixOperator(
            op = self.elasticity_model.assemble_partial_u_A_q_u(
                q_np = q.to_numpy().flatten()
            )
        )
    
    def clear_rhs_boundary_dofs(self, 
                                rhs: VectorArray,
                                flip: bool = False) -> VectorArray:
        
        assert isinstance(rhs, ListVectorArray)        
        for v in rhs.vectors:
            self.elasticity_model.clear_rhs_boundary_dofs(v.impl)

        if flip:
            return self.flip_vector_array(rhs)
        else:
            return rhs

    def flip_vector_array(self, vector_array: VectorArray) -> VectorArray:
        assert isinstance(vector_array, ListVectorArray)
        vector_array = vector_array.space.make_array(vector_array.vectors[::-1])
        return vector_array

    def get_translation_operator(self) -> Operator | None:
        if self.elasticity_model.m_has_translation_operator:
            return SparseMatrixOperator(
                op = self.elasticity_model.get_translation_operator()
            )
        else:
            return None
        
    def get_parameteric_operator(self, q: VectorArray) -> Operator:
        assert q in self.Q
        assert len(q) == 1
        
        return SparseMatrixOperator(
            op = self.elasticity_model.assemble_A_q(
                q_np = q.to_numpy().flatten(),
                linear_part_only = True,
            )
        )

# class ElasticitiyFOMEvaluatorB(FOMEvaluatorB):
#     def __init__(self,
#                  source : VectorSpace,
#                  range : VectorSpace,
#                  Q : VectorSpace,
#                  V : VectorSpace,
#                  elasticity_model: ElasticityModel):
        
#         super().__init__(source, range, Q, V)

#         self._Q = DealIIVectorSpace(Q.dim)
#         self.elasticity_model = elasticity_model

#     def __call__(self, 
#                  u: ListVectorArray,
#                  time_step: int) -> B_u:
#         assert u in self.V
#         # TODO Check how this function can be vectorized
#         assert len(u) == 1
#         assert isinstance(u, ListVectorArray)
        
#         self.elasticity_model.assemble_partial_q_A_q_u(
#             u.vectors[0].real_part.impl,
#             time_step = time_step
#         )

#         B_u_op = DealIIBaseOperator(op = self.elasticity_model.get_partial_q_A_q_u(
#             time_step = time_step
#         ))


#         def _B_u(d: NumpyVectorArray) -> pd2.Vector:
#             # TODO Move parameter space handling to C++ and use pd2.Vector
#             d_ = self._Q.from_numpy(d.to_numpy())
#             return B_u_op.apply(d_).vectors[0].real_part.impl
            
#         def _B_u_ad(p: ListVectorArray) -> np.ndarray:
#             return B_u_op.apply_adjoint(p).to_numpy()

#         return SimpleNamespace(B_u=_B_u, B_u_ad=_B_u_ad)