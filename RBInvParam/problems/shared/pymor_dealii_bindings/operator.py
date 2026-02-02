# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from typing import Optional, Type, Dict, Literal, Any

import pymor_dealii_bindings as pd2

from .vectorarray import DealIIVectorSpace
#from pymor.operators.list import LinearComplexifiedListVectorArrayOperatorBase
from pymor.operators.list import ListVectorArrayOperatorBase
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.vectorarrays.list import NumpyListVectorSpace

#####################################################################

class DealIIMatrixOperator(ListVectorArrayOperatorBase):
    """Wraps a dealII matrix as an |Operator|."""
    linear = True

    def __init__(self, matrix, name=None):
        self.source = DealIIVectorSpace(matrix.n())
        self.range = DealIIVectorSpace(matrix.m())
        #self.solver = pd2.SparseILU()
        # self._solver = None
        # self._solver_initialized = False
        self.__auto_init(locals())

    def _apply_one_vector(self, u, mu=None, prepare_data=None):
        r = self.range.zero_vector()
        self.matrix.vmult(r.impl, u.impl)
        return r

    def _apply_inverse_one_vector(
        self, v, mu=None, initial_guess=None, least_squares=False, prepare_data=None
    ):
        if least_squares:
            raise NotImplementedError
        r = self.source.zero_vector()
        self.matrix.cg_solve(r.impl, v.impl)
        return r

    def _apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        r = self.source.zero_vector()
        self.matrix.Tvmult(r.impl, v.impl)
        return r

    def _apply_inverse_adjoint_one_vector(self, u, mu=None, initial_guess=None, least_squares=False,
                                          prepare_data=None):
        raise NotImplementedError

    def _assemble_lincomb(
        self,
        operators,
        coefficients,
        identity_shift=0.0,
        solver_options=None,
        name=None,
    ):
        if not all(isinstance(op, (DealIIMatrixOperator)) for op in operators):
            return None
        if identity_shift != 0.0:
            return None
        assert not solver_options  # linear solver is not yet configurable

        matrix = pd2.SparseMatrix(operators[0].matrix.get_sparsity_pattern())
        matrix.copy_from(operators[0].matrix)
        matrix *= coefficients[0]
        for op, c in zip(operators[1:], coefficients[1:]):
            matrix.add(c, op.matrix)
        return DealIIMatrixOperator(matrix, name=name)

class DealIISymmetricMatrixOperator(DealIIMatrixOperator):
    """Wraps a symmetric deal.II matrix as an |Operator|.

    For a symmetric real matrix, the adjoint equals the operator itself, so
    apply_adjoint is the same as apply.
    """

    def _real_apply_inverse_adjoint_one_vector(self, u, mu=None, initial_guess=None, least_squares=False,
                                               prepare_data=None):
        if least_squares:
            raise NotImplementedError
        r = self.source.real_zero_vector()
        self.matrix.cg_solve(r.impl, u.impl)
        return r


    def _assemble_lincomb(
        self,
        operators,
        coefficients,
        identity_shift=0.0,
        solver_options=None,
        name=None,
    ):
        # Keep the symmetric type only if all operands are symmetric and no shift is applied.
        if not all(isinstance(op, DealIISymmetricMatrixOperator) for op in operators):
            return None
        if identity_shift != 0.0:
            return None
        assert not solver_options

        matrix = pd2.SparseMatrix(operators[0].matrix.get_sparsity_pattern())
        matrix.copy_from(operators[0].matrix)
        matrix *= coefficients[0]
        for op, c in zip(operators[1:], coefficients[1:]):
            matrix.add(c, op.matrix)
        return DealIISymmetricMatrixOperator(matrix, name=name)

#####################################################################

SpaceKind = Literal["dealii", "numpy"]

def _wrap_dealii_operator(
    op: pd2.BaseOperator,
    name: Optional[str] = None,
    source_space: SpaceKind = "dealii",
    range_space: SpaceKind = "dealii",
):
    """Wrap a deal.II operator into the corresponding wrapper type."""
    if not isinstance(op, pd2.BaseOperator):
        raise TypeError(f"op must be pd2.BaseOperator, got {type(op).__name__}")

    wrappers: Dict[Type[pd2.BaseOperator], Type[DealIIBaseOperator]] = {
        pd2.SparseMatrixOperator: SparseMatrixOperator,
        pd2.FullMatrixOperator: FullMatrixOperator,
        pd2.BaseOperator: DealIIBaseOperator,  # fallback
    }

    # Order matters: Sparse/Full should match before BaseOperator
    for op_type in (pd2.SparseMatrixOperator, pd2.FullMatrixOperator, pd2.BaseOperator):
        if isinstance(op, op_type):
            wrapper_cls = wrappers[op_type]
            return wrapper_cls(op, name=name, source_space=source_space, range_space=range_space)

    # Practically unreachable due to BaseOperator check above, but keep for completeness.
    raise TypeError(
        f"Unsupported deal.II operator type: {type(op).__name__}. "
        f"Expected one of: {', '.join(t.__name__ for t in wrappers)}."
    )

# TODO Add protocols for adapters to assert Runtime correctness.
class _IdentityAdapter:
    """No conversion; exposed vectors are already deal.II vectors."""
    def __init__(self, space):
        self.space = space

    def to_native(self, u):
        return u

    def from_native(self, v):
        return v


class _NumpyToDealIIAdapter:
    """Expose numpy vectors, but compute with deal.II vectors internally."""
    def __init__(self, np_space, dealii_space):
        self.space = np_space
        self._dealii_space = dealii_space

    def to_native(self, u):
        return self._dealii_space.vector_from_numpy(u.to_numpy())

    def from_native(self, v):
        return self.space.vector_from_numpy(v.to_numpy())


class DealIIBaseOperator(ListVectorArrayOperatorBase):
    def __init__(
        self,
        op: pd2.BaseOperator,
        name: Optional[str] = None,
        source_space: SpaceKind = "dealii",
        range_space: SpaceKind = "dealii",
    ):
        if not isinstance(op, pd2.BaseOperator):
            raise TypeError(f"op must be pd2.BaseOperator, got {type(op).__name__}")

        self.op = op
        self.linear = bool(op.linear)

        # Store so jacobians can preserve config
        self.source_space: SpaceKind = source_space
        self.range_space: SpaceKind = range_space

        self._native_source = DealIIVectorSpace(op.dim_source())
        self._native_range = DealIIVectorSpace(op.dim_range())

        self._source_adapter = self._make_adapter(self.source_space, self._native_source)
        self._range_adapter = self._make_adapter(self.range_space, self._native_range)

        self.source = self._source_adapter.space
        self.range = self._range_adapter.space

        self.__auto_init(locals())

    @staticmethod
    def _make_adapter(kind: SpaceKind, native_space):
        if kind == "dealii":
            return _IdentityAdapter(native_space)
        if kind == "numpy":
            np_space = NumpyListVectorSpace(dim=native_space.dim)
            return _NumpyToDealIIAdapter(np_space, native_space)
        raise ValueError(f"Unknown space kind: {kind!r} (expected 'dealii' or 'numpy')")

    def _apply_one_vector(self, u, mu=None, prepare_data=None):
        u_native = self._source_adapter.to_native(u)
        r_native = self._native_range.zero_vector()
        self.op.apply(r_native.impl, u_native.impl)
        return self._range_adapter.from_native(r_native)

    def _apply_inverse_one_vector(
        self, v, mu=None, initial_guess=None, least_squares: bool = False, prepare_data=None
    ):
        if least_squares:
            raise NotImplementedError("least_squares inverse is not implemented.")

        v_native = self._range_adapter.to_native(v)
        r_native = self._native_source.zero_vector()
        self.op.apply_inverse(r_native.impl, v_native.impl)
        return self._source_adapter.from_native(r_native)

    def _apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        v_native = self._range_adapter.to_native(v)
        r_native = self._native_source.zero_vector()
        self.op.apply_adjoint(r_native.impl, v_native.impl)
        return self._source_adapter.from_native(r_native)

    def _apply_inverse_adjoint_one_vector(
        self, v, mu=None, initial_guess=None, least_squares: bool = False, prepare_data=None
    ):
        if least_squares:
            raise NotImplementedError("least_squares inverse adjoint is not implemented.")

        v_native = self._source_adapter.to_native(v)
        r_native = self._native_range.zero_vector()
        self.op.apply_inverse_adjoint(r_native.impl, v_native.impl)
        return self._range_adapter.from_native(r_native)

    def jacobian(self, U, mu=None):
        if U not in self.source:
            raise ValueError("U must be an element of self.source.")
        if len(U) != 1:
            raise ValueError("jacobian expects a single-vector VectorArray (len(U) == 1).")

        u_native = self._source_adapter.to_native(U.vectors[0])
        J_native = self.op.jacobian(u_native.impl)

        return _wrap_dealii_operator(
            J_native,
            source_space=self.source_space,
            range_space=self.range_space,
        )

class SparseMatrixOperator(DealIIBaseOperator):
    def __init__(
        self,
        op: pd2.SparseMatrixOperator,
        name: Optional[str] = None,
        source_space: SpaceKind = "dealii",
        range_space: SpaceKind = "dealii",
    ):
        if not isinstance(op, pd2.SparseMatrixOperator):
            raise TypeError(f"op must be pd2.SparseMatrixOperator, got {type(op).__name__}")
        super().__init__(op, name=name, source_space=source_space, range_space=range_space)

    # IMPORTANT: remove jacobian override; base implementation is adapter-safe and preserves config

    def _assemble_lincomb(
        self,
        operators,
        coefficients,
        identity_shift: float = 0.0,
        solver_options=None,
        name: Optional[str] = None,
    ):
        if not all(isinstance(op, SparseMatrixOperator) for op in operators):
            return None
        if identity_shift != 0.0:
            return None
        if solver_options:
            raise NotImplementedError("solver_options are not yet configurable")

        sp = operators[0].op.get_matrix().get_sparsity_pattern()
        matrix = pd2.SparseMatrix(sp)
        matrix.copy_from(operators[0].op.get_matrix())
        matrix *= coefficients[0]

        for _op, c in zip(operators[1:], coefficients[1:]):
            matrix.add(c, _op.op.get_matrix())

        return SparseMatrixOperator(
            pd2.SparseMatrixOperator(matrix=matrix),
            name=name,
            source_space=self.source_space,
            range_space=self.range_space,
        )


class FullMatrixOperator(DealIIBaseOperator):
    def __init__(
        self,
        op: pd2.FullMatrixOperator,
        name: Optional[str] = None,
        source_space: SpaceKind = "dealii",
        range_space: SpaceKind = "dealii",
    ):
        if not isinstance(op, pd2.FullMatrixOperator):
            raise TypeError(f"op must be pd2.FullMatrixOperator, got {type(op).__name__}")
        super().__init__(op, name=name, source_space=source_space, range_space=range_space)

    # IMPORTANT: remove jacobian override; base implementation is adapter-safe and preserves config

# class NumpyDealIIFullMatrixOperator(FullMatrixOperator):
#     def __init__(self, op, name=None):
#         assert isinstance(op, pd2.FullMatrixOperator)
#         super().__init__(op)

#         self.dealii_source = DealIIVectorSpace(dim=self.op.dim_source())
#         self.dealii_range = DealIIVectorSpace(dim=self.op.dim_range())

#         self.np_source = NumpyListVectorSpace(dim=self.dealii_source.dim)
#         self.np_range = NumpyListVectorSpace(dim=self.dealii_range.dim)

#         self.source = self.np_source
#         self.range = self.dealii_range
        

#     def _apply_one_vector(self, u, mu=None, prepare_data=None):
#         u = self.dealii_source.vector_from_numpy(u.to_numpy())
#         r = self.range.zero_vector()
#         self.op.apply(r.impl, u.impl)
#         return r

#     def _apply_inverse_one_vector(
#         self, v, mu=None, initial_guess=None, least_squares=False, prepare_data=None
#     ):  
#         if least_squares:
#             raise NotImplementedError
        
#         r = self.dealii_source.zero_vector()
#         self.op.apply_inverse(r.impl, v.impl)

#         return r.to_numpy()
    
#     def _apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
#         r = self.dealii_source.zero_vector()
#         self.op.apply_adjoint(r.impl, v.impl)

#         return r.to_numpy()

#     def _apply_inverse_adjoint_one_vector(self, v, mu=None, initial_guess=None, least_squares=False,
#                                           prepare_data=None):
#         if least_squares:
#             raise NotImplementedError
#         r = self.source.zero_vector()
#         self.op.apply_inverse_adjoint(r.impl, v.impl)

#         return r