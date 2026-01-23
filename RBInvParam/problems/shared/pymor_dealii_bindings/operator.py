# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

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

def wrap_dealii_operator(op, *, linear=False):
    if isinstance(op, pd2.SparseMatrixOperator):
        return SparseMatrixOperator(op=op)
    elif isinstance(op, pd2.FullMatrixOperator):
        return FullMatrixOperator(op=op)
    elif isinstance(op, pd2.BaseOperator):
        return DealIIBaseOperator(op=op, linear=linear)
    else:
        raise TypeError(f"Unsupported deal.II operator type: {type(op)}")


class DealIIBaseOperator(ListVectorArrayOperatorBase):
    linear = False

    def __init__(self, op, name=None, linear=False):
        assert isinstance(op, pd2.BaseOperator)

        self.op = op
        self.source = DealIIVectorSpace(op.dim_source())
        self.range = DealIIVectorSpace(op.dim_range())
        self.linear = linear

        self.__auto_init(locals())

    def _apply_one_vector(self, u, mu=None, prepare_data=None):
        r = self.range.zero_vector()
        self.op.apply(r.impl, u.impl)
        return r

    def _apply_inverse_one_vector(
        self, v, mu=None, initial_guess=None, least_squares=False, prepare_data=None
    ):
        if least_squares:
            raise NotImplementedError
        r = self.source.zero_vector()
        self.op.apply_inverse(r.impl, v.impl)
        return r

    def _apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        r = self.source.zero_vector()
        self.op.apply_adjoint(r.impl, v.impl)
        return r

    def _apply_inverse_adjoint_one_vector(self, v, mu=None, initial_guess=None, least_squares=False,
                                          prepare_data=None):
        if least_squares:
            raise NotImplementedError
        r = self.source.zero_vector()
        self.op.apply_inverse_adjoint(r.impl, v.impl)
        return r

    def jacobian(self, U, mu=None):
        assert U in self.source
        assert len(U) == 1

        # return SparseMatrixOperator(
        #     op=self.op.jacobian(U.vectors[0].impl)
        # )

        return wrap_dealii_operator(
            self.op.jacobian(U.vectors[0].impl),
            linear=True,
        )

class SparseMatrixOperator(DealIIBaseOperator):    
    linear = True

    def __init__(self, op, name=None):
        assert isinstance(op, pd2.SparseMatrixOperator)
        super().__init__(op)
    
    def jacobian(self, U, mu=None):
        assert U in self.source
        assert len(U) == 1

        return SparseMatrixOperator(
            op = self.op.jacobian(U.vectors[0].impl)
        )
        
    def _assemble_lincomb(
        self,
        operators,
        coefficients,
        identity_shift=0.0,
        solver_options=None,
        name=None,
    ):

        if not all(isinstance(op, (SparseMatrixOperator)) for op in operators):
            return None
        if identity_shift != 0.0:
            return None
        assert not solver_options  # linear solver is not yet configurable

        matrix = pd2.SparseMatrix(operators[0].op.get_matrix().get_sparsity_pattern())
        matrix.copy_from(operators[0].op.get_matrix())
        matrix *= coefficients[0]
        for _op, c in zip(operators[1:], coefficients[1:]):
            matrix.add(c, _op.op.get_matrix())

        return SparseMatrixOperator(pd2.SparseMatrixOperator(matrix=matrix), name=name)

class FullMatrixOperator(DealIIBaseOperator):    
    linear = True

    def __init__(self, op, name=None):
        assert isinstance(op, pd2.FullMatrixOperator)
        super().__init__(op)
    
    def jacobian(self, U, mu=None):
        assert U in self.source
        assert len(U) == 1

        return FullMatrixOperator(
            op = self.op.jacobian(U.vectors[0].impl)
        )

class NumpyDealIIFullMatrixOperator(FullMatrixOperator):
    def __init__(self, op, name=None):
        assert isinstance(op, pd2.FullMatrixOperator)
        super().__init__(op)

        self.dealii_source = DealIIVectorSpace(dim=self.op.dim_source())
        self.dealii_range = DealIIVectorSpace(dim=self.op.dim_range())

        self.np_source = NumpyListVectorSpace(dim=self.dealii_source.dim)
        self.np_range = NumpyListVectorSpace(dim=self.dealii_range.dim)

        self.source = self.np_source
        self.range = self.dealii_range
        

    def _apply_one_vector(self, u, mu=None, prepare_data=None):
        u = self.dealii_source.vector_from_numpy(u.to_numpy())
        r = self.range.zero_vector()
        self.op.apply(r.impl, u.impl)
        return r

    def _apply_inverse_one_vector(
        self, v, mu=None, initial_guess=None, least_squares=False, prepare_data=None
    ):  
        if least_squares:
            raise NotImplementedError
        
        r = self.dealii_source.zero_vector()
        self.op.apply_inverse(r.impl, v.impl)

        return r.to_numpy()
    
    def _apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        r = self.dealii_source.zero_vector()
        self.op.apply_adjoint(r.impl, v.impl)

        return r.to_numpy()

    def _apply_inverse_adjoint_one_vector(self, v, mu=None, initial_guess=None, least_squares=False,
                                          prepare_data=None):
        if least_squares:
            raise NotImplementedError
        r = self.source.zero_vector()
        self.op.apply_inverse_adjoint(r.impl, v.impl)

        return r