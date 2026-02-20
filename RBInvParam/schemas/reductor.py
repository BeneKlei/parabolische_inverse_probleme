from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Mapping, Iterable, List
from enum import Enum

from RBInvParam.schemas.utils import *

# ----------------------------
# Reductor config (typed)
# ----------------------------

class LinearizationMethod(Enum):
    DEIM = "DEIM"


@dataclass(frozen=True)
class ErrorEstimatorTypes:
    state: Any
    adjoint: Any
    objective: Any

    def validate(self, *, where: str = "ErrorEstimatorTypes") -> None:
        # Keep this light since your enums live elsewhere; just ensure presence.
        if self.state is None:
            raise ValueError(f"{where}.state must not be None")
        if self.adjoint is None:
            raise ValueError(f"{where}.adjoint must not be None")
        if self.objective is None:
            raise ValueError(f"{where}.objective must not be None")

    @classmethod
    def defaults(cls) -> "ErrorEstimatorTypes":
        # No safe generic defaults here because these are project-specific enums.
        # Force user to provide them (or keep using your previous defaults upstream).
        raise RuntimeError("No defaults for ErrorEstimatorTypes; provide explicit values.")

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, where: str
    ) -> "ErrorEstimatorTypes":
        unknown_keys(data, cls.__dataclass_fields__.keys(), where=where)
        require(data, ["state", "adjoint", "objective"], where=where)

        cfg = cls(
            state=data["state"],
            adjoint=data["adjoint"],
            objective=data["objective"],
        )
        cfg.validate(where=where)
        return cfg

@dataclass(frozen=True)
class InstationaryReductorConfig:
    error_estimators: ErrorEstimatorTypes
    check_orthonormality: bool = True
    check_tol: float = 1e-9
    residual_image_basis_mode: str = "none"
    offline_parallel: bool = False
    use_adjoint_space: bool = False
    linearization_method: Any = None  # LinearizationMethod, but keep Any to avoid circular imports

    @classmethod
    def defaults(cls) -> "InstationaryReductorConfig":
        # NOTE: we cannot choose safe defaults for error_estimators here.
        raise RuntimeError(
            "InstationaryReductorConfig.defaults() is not available because "
            "error_estimators must be provided explicitly."
        )

    def validate(self, active_bases: Optional[List[str]] = None, *, where: str = "reductor") -> None:
        if self.check_tol <= 0:
            raise ValueError(f"{where}.check_tol must be > 0")
        if self.residual_image_basis_mode not in ("none",):
            raise ValueError(
                f"{where}.residual_image_basis_mode must be one of ('none',), "
                f"got {self.residual_image_basis_mode!r}"
            )

        if active_bases is not None:
            if self.use_adjoint_space and "adjoint_basis" in active_bases:
                raise ValueError(
                    f"{where}: use_adjoint_space=True requires 'adjoint_basis' not in active_bases"
                )

        # nested validate
        self.error_estimators.validate(where=f"{where}.error_estimators")

    @classmethod
    def from_dict(
        cls,
        data: Optional[Dict[str, Any]],
        *,
        where: str,
        active_bases: Optional[List[str]] = None,
        default_linearization_method: Optional[Any] = None,  # LinearizationMethod.DEIM typically
    ) -> "InstationaryReductorConfig":
        """
        Expected schema (matches your current optimizer_parameter['reductor']):

            {
                'use_adjoint_space': False,
                'offline_parallel': True,
                'error_estimator_types': {
                    'state': ...,
                    'adjoint': ...,
                    'objective': ...,
                },
                'check_orthonormality': True,
                'check_tol': 1e-9,
                'linearization_method': LinearizationMethod.DEIM,
                'residual_image_basis_mode': 'none',   # optional
            }

        Notes:
          - Accepts legacy key 'error_estimator_types' and maps it to error_estimators.
          - Validates unknown keys.
          - Validates against active_bases if provided.
        """
        # Provide a minimal base; we *must* fill error_estimators from data.
        if not data:
            raise KeyError(f"Missing keys in {where}: ['error_estimator_types']")

        allowed = {
            "error_estimator_types",          # legacy key
            "error_estimators",               # new key (optional support)
            "check_orthonormality",
            "check_tol",
            "residual_image_basis_mode",
            "offline_parallel",
            "use_adjoint_space",
            "linearization_method",
        }
        unknown_keys(data, allowed, where=where)

        # support both keys, prefer "error_estimators" if present
        if "error_estimators" in data:
            ee_raw = data["error_estimators"]
            ee_where = f"{where}['error_estimators']"
        else:
            require(data, ["error_estimator_types"], where=where)
            ee_raw = data["error_estimator_types"]
            ee_where = f"{where}['error_estimator_types']"

        if not isinstance(ee_raw, Mapping):
            raise TypeError(f"{ee_where} must be a mapping/dict")

        ee = ErrorEstimatorTypes.from_dict(dict(ee_raw), where=ee_where)

        # default linearization method handling
        lin_method = data.get("linearization_method", default_linearization_method)
        if lin_method is None:
            # If you want to enforce always-provided, change this to raise.
            # Keeping it permissive to match your existing setup.
            lin_method = default_linearization_method

        cfg = cls(
            error_estimators=ee,
            check_orthonormality=bool(data.get("check_orthonormality", True)),
            check_tol=float(data.get("check_tol", 1e-9)),
            residual_image_basis_mode=str(data.get("residual_image_basis_mode", "none")),
            offline_parallel=bool(data.get("offline_parallel", False)),
            use_adjoint_space=bool(data.get("use_adjoint_space", False)),
            linearization_method=lin_method,
        )
        cfg.validate(active_bases=active_bases, where=where)
        return cfg