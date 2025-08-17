import pymor_dealii_bindings as pd2
from types import SimpleNamespace

import pymor.vectorarrays as VectorArray

from pymor.vectorarrays.interface import VectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.list import ListVectorArray
from pymor.vectorarrays.numpy import NumpyVectorArray


from RBInvParam.evaluators import FOMEvaluatorA, FOMEvaluatorB, BU
from RBInvParam.problems.elasticity.material_model import MaterialModel
from RBInvParam.problems.elasticity.pymor_dealii_bindings.operator import DealIIMatrixOperator



class ElasticitiyFOMEvaluatorA(FOMEvaluatorA):
    def __init__(self,
                 material_model: MaterialModel,
                 source : VectorSpace,
                 range : VectorSpace,
                 Q : VectorSpace):
        
        assert source == range
        self.Q = Q
        self.source = source
        self.range = range
        self.material_model = material_model
        

        self.system_matrix = pd2.SparseMatrix()
        self.sparsity_pattern = self.material_model.sparsity_pattern()
    
    def __call__(self, q: VectorArray) -> DealIIMatrixOperator:
        assert q in self.Q
        
        self.system_matrix.reinit(self.sparsity_pattern)
        self.material_model.m_q[:] = q.to_numpy()
        self.material_model.assemble_system_matrix(self.system_matrix)
        return DealIIMatrixOperator(self.system_matrix)
    
    def clear_rhs_boundary_dofs(self, 
                                rhs: ListVectorArray,
                                flip: bool = False) -> ListVectorArray:
        
        assert isinstance(rhs, ListVectorArray)
        for v in rhs.vectors:
            self.material_model.clear_rhs_boundary_dofs(v.real_part.impl)
        

        if flip:
            return self.flip_vector_array(rhs)
        else:
            return rhs

    def flip_vector_array(self, vector_array: ListVectorArray) -> ListVectorArray:
        assert isinstance(vector_array, ListVectorArray)
        return vector_array.space.make_array(vector_array.vectors[::-1])

class ElasticitiyFOMEvaluatorB(FOMEvaluatorB):
    def __init__(self,
                 material_model: MaterialModel,
                 source : VectorSpace,
                 range : VectorSpace,
                 Q : VectorSpace,
                 V : VectorSpace):
        
        self.Q = Q
        self.V = V
        self.source = source
        self.range = range
        self.material_model = material_model

    def __call__(self, u: ListVectorArray) -> BU:
        assert u in self.V
        # TODO Check how this function can be vectorized
        assert len(u) == 1
        assert isinstance(u, ListVectorArray)

            
        B_u_mat = pd2.FullMatrix(self.V.dim, self.Q.dim)
        self.material_model.assemble_system_matrix_derivative(B_u_mat, u.vectors[0].real_part.impl)
        B_u_op = DealIIMatrixOperator(matrix = B_u_mat)

        def _B_u(d: ListVectorArray) -> NumpyVectorArray:
            return self.Q.make_array(B_u_op.apply(d).to_numpy())
            
        def _B_u_ad(p: ListVectorArray) -> NumpyVectorArray:
            return self.Q.make_array(B_u_op.apply_adjoint(p).to_numpy())

        return SimpleNamespace(B_u=_B_u, B_u_ad=_B_u_ad)