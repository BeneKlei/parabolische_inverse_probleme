import pymor_dealii_bindings as pd2

from pymor.vectorarrays.interface import VectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
import pymor.vectorarrays as VectorArray

from RBInvParam.evaluators import EvaluatorA, EvaluatorB, BU
from RBInvParam.problems.elasticity.material_model import MaterialModel
from RBInvParam.problems.elasticity.pymor_dealii_bindings.operator import DealIIMatrixOperator


class FOMEvaluatorA(EvaluatorA):
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
        
    
    def __call__(self, q: VectorArray) -> DealIIMatrixOperator:
        assert q in self.Q

        self.material_model.m_q[:] = q.to_numpy()
        M = pd2.SparseMatrix()
        M.reinit(self.material_model.sparsity_pattern())
        self.material_model.assemble_mass_matrix(M)
        return DealIIMatrixOperator(M)
       

