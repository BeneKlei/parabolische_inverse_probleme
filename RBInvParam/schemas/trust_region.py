from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray

# ----------------------------
# Result contract
# ----------------------------

@dataclass(frozen=True)
class TRCheckResult:
    tr_ok: bool
    model_insufficient: bool


# ----------------------------
# Context contract (Optimizer passes this)
# ----------------------------

@dataclass(frozen=True)
class TRContext:
    # error-type TRs
    objective: float = np.nan
    abs_error: Optional[float] = None

    # radius/step-type TRs
    center_q: Optional[VectorArray] = None
    current_q: Optional[VectorArray] = None
    product: Optional[Operator] = None

    # optional extras for custom TRs
    step_size: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


# ----------------------------
# Enum
# ----------------------------

class TRType(str, Enum):
    NONE = "none"
    RELATIVE_OBJECTIVE_ERROR = "relative_objective_error"
    RADIUS = "radius"


# ----------------------------
# Default config (for “eta-based” TRs)
# ----------------------------

@dataclass(frozen=True)
class TrustRegionConfig:
    eta_initial: float
    eta_min: float
    eta_max: float
    beta_1: float
    beta_2: float
    beta_3: float

    @classmethod
    def defaults(cls) -> "TrustRegionConfig":
        return cls(
            eta_initial=1.0,
            eta_min=0.0,
            eta_max=float("inf"),
            beta_1=0.25,
            beta_2=0.75,
            beta_3=0.5,
        )

    def validate(self) -> None:
        if self.eta_initial <= 0:
            raise ValueError("eta_initial must be > 0")
        if not (0 <= self.eta_min <= self.eta_max):
            raise ValueError("Require 0 <= eta_min <= eta_max")
        if self.beta_1 <= 0 or self.beta_2 <= 0 or self.beta_3 <= 0:
            raise ValueError("beta_1, beta_2, beta_3 must all be > 0")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TrustRegionConfig":
        base = cls.defaults()
        if not data:
            base.validate()
            return base

        valid_keys = set(cls.__dataclass_fields__.keys())
        unknown = set(data.keys()) - valid_keys
        if unknown:
            raise ValueError(f"Unknown TrustRegionConfig keys: {sorted(unknown)}")

        cfg = replace(
            base,
            eta_initial=float(data.get("eta_initial", base.eta_initial)),
            eta_min=float(data.get("eta_min", base.eta_min)),
            eta_max=float(data.get("eta_max", base.eta_max)),
            beta_1=float(data.get("beta_1", base.beta_1)),
            beta_2=float(data.get("beta_2", base.beta_2)),
            beta_3=float(data.get("beta_3", base.beta_3)),
        )
        cfg.validate()
        return cfg
