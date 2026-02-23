from typing import List

import numpy as np
import pymor_dealii_bindings as pd2
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import LincombOperator

from pymor.operators.constructions import LincombOperator

from RBInvParam.reduction.registry import register_reductor
from RBInvParam.reduction.default import DefaultIPReductor


@register_reductor("material_model")
class MaterialModelIPReductor(DefaultIPReductor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ---------------- C++ ROM projection caches ----------------
        # basis_name -> pd2.DenseBasis
        self._mm_cpp_basis = {}

        # basis_name -> last basis size seen by C++
        self._mm_cpp_r = {}

        # cache_key -> pd2.ReducedProjector
        self._mm_cpp_proj = {}

        # cache_key -> id of native operator list (detect operator changes)
        self._mm_cpp_ops_id = {}


    def delete_cached_operators(self,
                                targets: List[str] | str | None = None) -> None:
    
        assert targets is None
        # also reset C++ projection state
        self._mm_cpp_basis.clear()
        self._mm_cpp_r.clear()
        self._mm_cpp_proj.clear()
        self._mm_cpp_ops_id.clear()

    # ---------------------- internal helpers ----------------------

    def _mm_cache_init(self):
        # lazily create caches on first use
        if not hasattr(self, "_mm_cpp_basis"):
            self._mm_cpp_basis = {}        # basis_name -> pd2.DenseBasis
            self._mm_cpp_r = {}            # basis_name -> last basis length seen in C++
        if not hasattr(self, "_mm_cpp_proj"):
            self._mm_cpp_proj = {}         # cache_key -> pd2.ReducedProjector
        if not hasattr(self, "_mm_cpp_ops_id"):
            self._mm_cpp_ops_id = {}       # cache_key -> id(tuple(native_ops)) to detect operator list changes

    def _mm_cache_key(self, source_basis: str, range_basis: str) -> str:
        if source_basis == range_basis == "state_basis":
            return "A_r_state"
        if source_basis == range_basis == "adjoint_basis":
            return "A_r_adjoint"
        if (source_basis == "adjoint_basis") and (range_basis == "state_basis"):
            return "A_r_adjoint_state"
        raise ValueError(f"Unsupported (source_basis, range_basis)=({source_basis},{range_basis})")

    def _mm_get_or_rebuild_cpp_basis(self, basis_name: str, basis_va, n_full: int):
        """
        Ensure pd2.DenseBasis exists and append only new vectors.
        If basis shrank/reordered, rebuild from scratch.
        Returns (cpp_basis, k_added).
        """
        self._mm_cache_init()

        cpp_basis = self._mm_cpp_basis.get(basis_name)
        r_seen = self._mm_cpp_r.get(basis_name, 0)
        r_now = len(basis_va)

        if cpp_basis is None:
            # capacity heuristic: slightly above current size to reduce reallocations
            cap = max(64, r_now + 16)
            cpp_basis = pd2.DenseBasis(n_full=n_full, capacity=cap)
            self._mm_cpp_basis[basis_name] = cpp_basis
            r_seen = 0

        # If basis shrank or likely changed ordering, rebuild basis (safe default).
        # (Your dims_history suggests you extend monotonically; if that's guaranteed,
        # you can delete this branch.)
        if r_now < r_seen:
            cap = max(64, r_now + 16)
            cpp_basis = pd2.DenseBasis(n_full=n_full, capacity=cap)
            self._mm_cpp_basis[basis_name] = cpp_basis
            r_seen = 0

        if r_now == r_seen:
            return cpp_basis, 0

        # Append only new block
        W = basis_va[r_seen:r_now]

        W_np = W.to_numpy()  # often (k, n_full) for pyMOR
        if W_np.ndim != 2:
            raise RuntimeError("Expected basis block to_numpy() to return 2D array")

        k = r_now - r_seen

        # Make it (n_full, k)
        if W_np.shape[0] == k and W_np.shape[1] == n_full:
            W_np = W_np.T
        elif W_np.shape[0] == n_full and W_np.shape[1] == k:
            pass
        else:
            raise RuntimeError(f"Unexpected basis numpy shape {W_np.shape}, expected (k,n) or (n,k) with n={n_full}, k={k}")

        # Ensure float64; strides are handled in C++ so order is not critical
        W_np = np.asarray(W_np, dtype=np.float64, order="C")
        cpp_basis.append(W_np)

        self._mm_cpp_r[basis_name] = r_now
        return cpp_basis, k

    def _mm_native_sparse_ops(self, parameter_reduced_A: LincombOperator):
        """
        Extract native C++ pd2.SparseMatrixOperator objects from pyMOR operators list.
        """
        native_ops = []
        for op in parameter_reduced_A.operators:
            # Your wrappers store native operator at .op (see your code)
            native = getattr(op, "op", None)
            if native is None:
                raise TypeError(f"Operator {type(op)} has no .op; cannot use C++ projector")
            native_ops.append(native)
        return native_ops

    # ---------------------- public override ----------------------

    def project_A(self,
                  parameter_reduced_A: LincombOperator,
                  source_basis: str = "state_basis",
                  range_basis: str = "state_basis") -> LincombOperator:

        cache_key = self._mm_cache_key(source_basis, range_basis)

        # Only the same-basis case is supported by the current C++ projector
        # (single DenseBasis). For other cases, fall back to DefaultIPReductor implementation.
        if source_basis != range_basis:
            # You can add a 2-basis C++ projector later.
            return super().project_A(parameter_reduced_A, source_basis, range_basis)

        # If already cached and basis did not change, return it.
        # (We still might want to update cache if basis grew; handled below.)
        self._mm_cache_init()

        # Get current basis as pyMOR VectorArray
        basis_va = self._get_projection_basis(source_basis)
        if basis_va is None or len(basis_va) == 0:
            # nothing to project onto
            return super().project_A(parameter_reduced_A, source_basis, range_basis)

        # Extract native C++ operators
        native_ops = self._mm_native_sparse_ops(parameter_reduced_A)

        # Determine full dimension from operator
        # pd2.BaseOperator has dim_source()
        n_full = int(native_ops[0].dim_source())

        # Sync / append basis to C++ DenseBasis
        cpp_basis, k_added = self._mm_get_or_rebuild_cpp_basis(source_basis, basis_va, n_full=n_full)

        # Create or reuse projector
        proj = self._mm_cpp_proj.get(cache_key)

        # Detect operator-list changes: if parameter basis grew, you might have more operators now
        ops_id = id(tuple(native_ops))
        prev_ops_id = self._mm_cpp_ops_id.get(cache_key)

        if (proj is None) or (prev_ops_id != ops_id):
            # choose max_r: either config value or current basis size + headroom
            max_r = max(64, len(basis_va) + 16)

            proj = pd2.ReducedProjector(basis=cpp_basis, max_r=max_r, symmetric=False)
            proj.set_operators(native_ops)
            
            proj.project_full_all()

            self._mm_cpp_proj[cache_key] = proj
            self._mm_cpp_ops_id[cache_key] = ops_id
        else:
            # Basis grew -> incremental update
            if k_added > 0:
                proj.update_after_append(k_added)

        # Build LincombOperator of NumpyMatrixOperator (same output type you had)
        Ms = [proj.get(i) for i in range(len(native_ops))]
        operators = [NumpyMatrixOperator(matrix=M) for M in Ms]

        A_r = LincombOperator(operators=operators, coefficients=parameter_reduced_A.coefficients)
        self._cached_operators[cache_key] = A_r
        return A_r