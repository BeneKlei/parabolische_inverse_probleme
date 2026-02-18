# RBInvParam/utils/numerics.py
from __future__ import annotations
from dataclasses import dataclass
import sys
import numpy as np

MACHINE_EPS = sys.float_info.epsilon

@dataclass(frozen=True)
class ObjectivePolicy:
    eps: float = MACHINE_EPS
    neg_tol: float = 100 * MACHINE_EPS
    zero_tol: float = 100 * MACHINE_EPS

    # ---- objective is NEVER allowed to be nan ----
    def sanitize_objective(self, J: float, *, name: str = "objective") -> float:
        J = float(J)
        if np.isnan(J):
            raise ValueError(f"{name} is NaN")
        if np.isinf(J):
            raise ValueError(f"{name} is inf")
        if J < 0.0:
            if J >= -self.neg_tol:
                return 0.0
            raise ValueError(f"{name} must be >= 0, got {J:3.4e}")
        return J

    def sanitize_error(self, err: float | np.ndarray | None, *, name: str = "abs_error") -> float | np.ndarray:
        # None -> treat as "not available"
        if err is None:
            return np.nan

        # ---- scalar case ----
        if np.isscalar(err):
            err = float(err)

            if np.isnan(err):
                return np.nan  # allowed
            if np.isposinf(err):
                return np.inf
            if np.isneginf(err):
                raise ValueError(f"{name} is -inf (invalid)")

            if err < 0.0:
                if err >= -self.neg_tol:
                    return 0.0
                raise ValueError(f"{name} must be >= 0 or NaN, got {err:3.4e}")
            return err

        # ---- array case ----
        a = np.asarray(err, dtype=float)

        # NaNs allowed (leave them as-is)
        # +inf allowed; -inf invalid
        if np.isneginf(a).any():
            raise ValueError(f"{name} contains -inf (invalid)")

        # Only check negative entries (ignore NaNs)
        neg_mask = (a < 0.0) & ~np.isnan(a)

        if neg_mask.any():
            # if any negative entry is "too negative", raise
            too_neg = a[neg_mask] < -self.neg_tol
            if np.any(too_neg):
                mn = float(np.min(a[neg_mask]))
                raise ValueError(f"{name} must be >= 0 or NaN; min entry {mn:3.4e} < -neg_tol")

            # otherwise clamp tiny negatives to 0 (only those entries)
            a = a.copy()
            a[neg_mask] = 0.0

        return a

    def is_effectively_zero(self, x: float) -> bool:
        return abs(float(x)) <= self.zero_tol

    def rel_error(
        self,
        *,
        abs_error: float | np.ndarray | None,
        objective: float
    ) -> float | np.ndarray:
        """
        Relative error with NaN support, scalar or array.

        Scalar logic applied elementwise for arrays:
        - NaN error -> NaN
        - if objective ~ 0:
                error ~ 0 -> 0
                else -> inf
        - else -> error / objective
        """
        J = self.sanitize_objective(objective, name="objective")
        err = self.sanitize_error(abs_error, name="abs_error")

        # ---- scalar case ----
        if np.isscalar(err):
            if np.isnan(err):
                return np.nan

            if self.is_effectively_zero(J):
                return 0.0 if self.is_effectively_zero(err) else np.inf

            return err / J

        # ---- array case ----
        # err is ndarray
        err_arr = err

        # result array
        rel = np.empty_like(err_arr, dtype=float)

        # NaNs propagate
        nan_mask = np.isnan(err_arr)
        rel[nan_mask] = np.nan

        # objective effectively zero
        if self.is_effectively_zero(J):
            zero_mask = self.is_effectively_zero(err_arr)
            rel[~nan_mask & zero_mask] = 0.0
            rel[~nan_mask & ~zero_mask] = np.inf
            return rel

        # normal division
        rel[~nan_mask] = err_arr[~nan_mask] / J

        return rel
GLOBAL_OBJ_POLICY = ObjectivePolicy()
