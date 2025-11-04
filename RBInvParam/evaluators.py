import numpy as np
from typing import List
from abc import ABC, abstractmethod
from typing import Protocol, Callable, runtime_checkable
from types import SimpleNamespace

from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.interface import VectorSpace, VectorArray
from pymor.vectorarrays.numpy import NumpyVectorArray

from RBInvParam.utils.discretization import Struct, build_projection

class EvaluatorA(ABC):
    def __init__(self,
                source : VectorSpace,
                range : VectorSpace,
                Q : VectorSpace,
                parameter_names: List[str] | None,
                translation_operator: bool = False):
    
        assert source == range
        self.Q = Q
        self.source = source
        self.range = range
        self.parameter_names = parameter_names
        self.translation_operator = translation_operator

    @abstractmethod
    def __call__(self, q: VectorArray) -> Operator:
        pass

    def get_parameter_names(self) -> List[str] | None:
        return self.parameter_names
    
    @abstractmethod
    def get_translation_operator(self) -> Operator | None:
        pass

    @abstractmethod
    def get_parameteric_operator(self) -> Operator:
        pass

class EvaluatorB(ABC):
    def __init__(self,
                 source : VectorSpace,
                 range : VectorSpace,
                 Q : VectorSpace,
                 V : VectorSpace):
        
        self.Q = Q
        self.V = V
        self.source = source
        self.range = range

    @abstractmethod
    def __call__(self, u: VectorArray) -> Struct:
        pass

@runtime_checkable
class BU(Protocol):
    B_u: Callable[[VectorArray], VectorArray]
    B_u_ad: Callable[[VectorArray], VectorArray]

class FOMEvaluatorA(EvaluatorA):
    def __init__(self,
                 source : VectorSpace,
                 range : VectorSpace,
                 Q : VectorSpace, 
                 parameter_names: List[str] | None):
        
        super().__init__(source, range, Q, parameter_names)

    @abstractmethod
    def __call__(self, q: VectorArray) -> Operator:
        pass

    @abstractmethod
    def clear_rhs_boundary_dofs(self, 
                                rhs: VectorArray,
                                flip: bool = False) -> VectorArray:
        pass

    @abstractmethod
    def flip_vector_array(self, vector_array: VectorArray) -> VectorArray:
        pass


class FOMEvaluatorB(EvaluatorB):
    def __init__(self,
                 source : VectorSpace,
                 range : VectorSpace,
                 Q : VectorSpace,
                 V : VectorSpace):
        
        super().__init__(source, range, Q, V)
            

    @abstractmethod
    def __call__(self, u: VectorArray) -> Struct:
        pass

class ROMEvaluatorA(EvaluatorA):
    def __init__(self, 
                 source : VectorSpace,
                 range : VectorSpace,
                 Q : VectorSpace,                 
                 parameteric_operator: Operator,
                 translation_operator : Operator | None):

        assert parameteric_operator.parametric
        assert parameteric_operator.source == source
        assert parameteric_operator.range == range

        if translation_operator:
            assert not translation_operator.parametric
            assert parameteric_operator.source == source
            assert parameteric_operator.range == range
        
        self.parameters = parameteric_operator.parameters
        self.parameteric_operator = parameteric_operator
        self.translation_operator = translation_operator

        parameter_names = ['reduced_parameter']
        assert parameter_names
        super().__init__(source, range, Q, parameter_names, translation_operator)

    def __call__(self, q: VectorArray) -> NumpyMatrixOperator:
        assert q in self.Q
        # TODO Can _assemble_A_q be vectorized?
        assert len(q) == 1

        q_as_par = self.parameters.parse(q.to_numpy()[0])

        if self.translation_operator:
            return self.parameteric_operator.assemble(q_as_par) + self.translation_operator
        else:
            return self.parameteric_operator.assemble(q_as_par)

    def get_translation_operator(self) -> Operator | None:
        return self.translation_operator

    def get_parameteric_operator(self) -> Operator:
        return self.parameteric_operator
    

    def clear_rhs_boundary_dofs(self, 
                                rhs: VectorArray,
                                flip: bool = False) -> VectorArray:
        assert isinstance(rhs, NumpyVectorArray)
        
        if flip:
            return self.flip_vector_array(rhs)
        else:
           return rhs

    def flip_vector_array(self, vector_array: VectorArray) -> VectorArray:
        assert isinstance(vector_array, NumpyVectorArray)
        vector_array = vector_array.space.make_array(vector_array.to_numpy()[::-1])
        return vector_array
    
class ROMEvaluatorB(EvaluatorB):
    def __init__(self, 
                 source : VectorSpace,
                 range : VectorSpace,
                 Q : VectorSpace,
                 V : VectorSpace,
                 parameteric_operator: Operator,
                 translation_operator : Operator | None):

        assert parameteric_operator.parametric
        assert parameteric_operator.source == range
        assert parameteric_operator.range == V

        if translation_operator:
            assert not translation_operator.parametric
            assert parameteric_operator.source == range
            assert parameteric_operator.range == V
        
        self.parameters = parameteric_operator.parameters
        self.parameteric_operator = parameteric_operator
        self.translation_operator = translation_operator
    
        super().__init__(source = source,
                         range = range,
                         Q = Q,
                         V = V)

    def __call__(self, u: VectorArray, parameter_basis_idx: int) -> BU:
        assert u in self.V
        assert len(u) == 1
        if not self.parameteric_operator:
            raise NotImplementedError
        
        DoFs = self.range.dim
        ops = self.parameteric_operator.operators
        T = len(ops)

        B_u_mat = np.empty((T, DoFs))
        for i, op in enumerate(ops):
            B_u_mat[i] = op.apply_adjoint(u).to_numpy()[0, :]

        def _B_u(d: VectorArray) -> VectorArray:
            d_np = d.to_numpy()[0]          # shape (T,)
            out = np.einsum("ti,t->i", B_u_mat, d_np)  # (DoFs,)
            return out[None, :][0]     # (1, DoFs)

        def _B_u_ad(p: VectorArray) -> VectorArray:
            p_np = p.to_numpy()[0]          # shape (DoFs,)
            out = np.einsum("ti,i->t", B_u_mat, p_np)  # (T,)
            return out[None, :]     # (1, T)

        # Return a simple object that satisfies the BU protocol
        return SimpleNamespace(B_u=_B_u, B_u_ad=_B_u_ad)