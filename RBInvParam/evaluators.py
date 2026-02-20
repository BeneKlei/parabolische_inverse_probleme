import numpy as np
from typing import Tuple 
from abc import ABC, abstractmethod
from numbers import Number

from pymor.operators.interface import Operator
from pymor.operators.constructions import LincombOperator, ZeroOperator, VectorArrayOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.interface import VectorSpace, VectorArray
from pymor.vectorarrays.numpy import NumpyVectorArray

from RBInvParam.utils.discretization import Struct, build_projection


class InvalidAssemblyArgument(Exception):
    pass

class EvaluatorA(ABC):
    def __init__(self,
                source : VectorSpace,
                range : VectorSpace,
                Q : VectorSpace,
                A_affine: bool = False,
                A_q_linear: bool = False):

        assert source == range
        self.Q = Q
        self.source = source
        self.range = range        
        self.A_affine = A_affine
        self.A_q_linear = A_q_linear

    @abstractmethod
    def get_A_q(self, q: VectorArray) -> Operator:
        pass
    
    @abstractmethod
    def get_partial_u_A_q_u(self, q: VectorArray , u: VectorArray, A_q: Operator = None) -> Operator:
        pass
    
    @abstractmethod
    def get_partial_q_A_q_u(self, q: VectorArray , u: VectorArray) -> Operator:
        pass

    @abstractmethod
    def get_translation_operator(self) -> Operator | None:
        pass

    @abstractmethod
    def get_parameteric_operator(self) -> Operator:
        pass

class FOMEvaluatorA(EvaluatorA):
    def __init__(self,
                 source : VectorSpace,
                 range : VectorSpace,
                 Q : VectorSpace,
                 A_affine: bool = False,
                 A_q_linear: bool = False):
                
        super().__init__(
            source, 
            range, 
            Q, 
            A_affine, 
            A_q_linear
        )

    @abstractmethod
    def get_A_q(self, q: VectorArray) -> Operator:
        pass
    
    @abstractmethod
    def get_partial_u_A_q_u(self, q: VectorArray , u: VectorArray, A_q: Operator = None) -> Operator:
        pass

    @abstractmethod
    def get_partial_q_A_q_u(self, q: VectorArray , u: VectorArray) -> Operator:
        pass

    @abstractmethod
    def clear_rhs_boundary_dofs(self,
                                rhs: VectorArray,
                                flip: bool = False) -> VectorArray:
        pass

class ROMEvaluatorA(EvaluatorA):
    def __init__(self,
                 source : VectorSpace,
                 range : VectorSpace,
                 Q : VectorSpace,
                 parameteric_operator: NumpyMatrixOperator,
                 translation_operator : NumpyMatrixOperator | None):

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
        self.A_q_linear_op = True

        parameter_names = ['reduced_parameter']
        assert parameter_names
        super().__init__(source, range, Q, parameter_names, translation_operator)

    def get_A_q(self, q: VectorArray) -> Operator:
        assert q in self.Q
        assert len(q) == 1

        q_as_par = self.parameters.parse(q.to_numpy()[0])
        _parameteric_operator = self.parameteric_operator.assemble(q_as_par)
        
        assert isinstance(_parameteric_operator, NumpyMatrixOperator)
        matrix = _parameteric_operator.matrix

        if self.translation_operator:
            assert isinstance(self.translation_operator, NumpyMatrixOperator)
            matrix = matrix + self.translation_operator.matrix

        return NumpyMatrixOperator(
            matrix = matrix
       )

    def get_partial_u_A_q_u(self, q: VectorArray , u: VectorArray, A_q: Operator = None) -> Operator:
        assert q in self.Q
        assert len(q) == 1

        if u is not None:
            assert u in self.source
            assert len(u) == 1

        if A_q is None:
            A_q = self.get_A_q(q)

        assert isinstance(A_q, Operator)

        return A_q

    def get_partial_q_A_q_u(self, q: VectorArray , u: VectorArray) -> Operator:
        assert q in self.Q
        assert len(q) == 1

        assert u in self.source
        assert len(u) == 1

        if not self.parameteric_operator:
            raise NotImplementedError

        DoFs = self.range.dim
        ops = self.parameteric_operator.operators
        T = len(ops)

        partial_q_A_q_u_mat = np.empty((T, DoFs))
        for i, op in enumerate(ops):
            partial_q_A_q_u_mat[i] = op.apply_adjoint(u).to_numpy()[0, :]

        return NumpyMatrixOperator(
            matrix = partial_q_A_q_u_mat.T
        )

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

    def get_translation_operator(self) -> Operator | None:
        return self.translation_operator

    def get_parameteric_operator(self) -> Operator:
        return self.parameteric_operator


class EvaluatorLincomb(EvaluatorA):
    def __init__(self, op: LincombOperator, Q: VectorSpace, A_affine: bool = True):
        assert isinstance(op, LincombOperator)

        linear_op, constant_op = self._split_affine_lincomb(op, A_affine=A_affine)

        self.op = op
        self.linear_op = linear_op
        self.constant_op = constant_op

        super().__init__(
            Q=Q,
            source=self.op.source,
            range=self.op.range,
            A_affine=A_affine,
            A_q_linear=self.op.linear,
        )

    def _split_affine_lincomb(
        self, op: LincombOperator, *, A_affine: bool
    ) -> Tuple[LincombOperator, Operator]:
        """Return (linear_op, constant_op) where constant_op is either ZeroOperator or exactly one constant summand."""
        is_const = lambda c: isinstance(c, Number) or getattr(c, "is_constant", False) or (
            hasattr(c, "parameters") and len(c.parameters) == 0
        )

        const_idx = [i for i, c in enumerate(op.coefficients) if is_const(c)]

        if not A_affine:
            # no constant part allowed
            assert len(const_idx) == 0, f"A_affine=False but found {len(const_idx)} constant term(s)"
            return op, ZeroOperator(op.source, op.range)

        # affine allowed: require exactly one constant summand
        assert len(const_idx) == 1, f"Expected exactly one constant term, got {len(const_idx)}"
        i = const_idx[0]

        constant_op = LincombOperator([op.operators[i]], [op.coefficients[i]])
        ops = [o for j, o in enumerate(op.operators) if j != i]
        coefs = [c for j, c in enumerate(op.coefficients) if j != i]
        linear_op = LincombOperator(ops, coefs) if ops else ZeroOperator(op.source, op.range)

        return linear_op, constant_op

    def get_A_q(self, q: VectorArray) -> Operator:
        assert q in self.Q
        assert len(q) == 1

        q_as_par = self.op.parameters.parse(q.to_numpy()[0])
        return self.op.assemble(q_as_par)
    
    def get_partial_u_A_q_u(self, q: VectorArray , u: VectorArray, A_q: Operator = None) -> Operator:
        assert q in self.Q
        assert len(q) == 1
        if u is not None:
            assert u in self.source
            assert len(u) == 1

        if A_q is None:
            A_q = self.get_A_q(q)

        try:
            return A_q.jacobian(U=u) 
        except:
            raise InvalidAssemblyArgument

    def get_partial_q_A_q_u(self, q: VectorArray , u: VectorArray) -> Operator:
        assert q in self.Q
        assert len(q) == 1

        assert u in self.source
        assert len(u) == 1

        ops = self.linear_op.operators
        partial_q_A_q_u_array = self.range.empty(reserve=len(ops))

        for op in ops:
            partial_q_A_q_u_array.append(op.apply(u))
        
        return VectorArrayOperator(
            array = partial_q_A_q_u_array
        )
    
    def clear_rhs_boundary_dofs(self,
                                rhs: VectorArray,
                                flip: bool = False) -> VectorArray:
        if flip:
            return rhs[::-1]
        else:
           return rhs
        
    def get_translation_operator(self) -> Operator:
        return self.constant_op.assemble()
    
    def get_parameteric_operator(self, q: VectorArray) -> Operator:
        assert q in self.Q
        assert len(q) == 1

        q_as_par = self.op.parameters.parse(q.to_numpy()[0])
        return self.linear_op.assemble(q_as_par)