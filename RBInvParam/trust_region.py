# trust_region.py  (FULL refactor)

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Optional, Type

import numpy as np

from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray

from RBInvParam.utils.logger import get_default_logger
from RBInvParam.products import BochnerProductOperator
from RBInvParam.optimizer.numerics import GLOBAL_OBJ_POLICY as OBJ

MACHINE_EPS = sys.float_info.epsilon

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


# ----------------------------
# Base class + registry
# ----------------------------

class TR(ABC):
    """
    Base trust-region controller.

    Key points:
      - Optimizer always calls check(ctx: TRContext)
      - TR decides what it needs (objective error? radius?)
      - eta/betas stored in config; only eta is mutable state
    """
    requires_objective_error: bool = True
    _registry: Dict[TRType, Type["TR"]] = {}

    def __init_subclass__(cls, *, tr_type: Optional[TRType] = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if tr_type is not None:
            TR._registry[tr_type] = cls

    # subclasses can override this to support custom configs
    @classmethod
    def parse_config(cls, config_dict: Optional[Dict[str, Any]]):
        return TrustRegionConfig.from_dict(config_dict)

    def __init__(self, config: Any, logger: Optional[logging.Logger] = None):
        # config may be TrustRegionConfig or a subclass-specific config
        if hasattr(config, "validate"):
            config.validate()

        self.config = config
        self._eta = float(getattr(config, "eta_initial", 1.0))

        self._logger = logger or get_default_logger(self.__class__.__name__)
        self._logger.setLevel(logging.DEBUG)
        self._logger.debug("Setting up %s", self.__class__.__name__)

    @classmethod
    def from_type(
        cls,
        tr_type: TRType,
        config_dict: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ) -> "TR":
        try:
            tr_class = cls._registry[tr_type]
        except KeyError:
            raise ValueError(f"No TR registered for type {tr_type}")

        cfg = tr_class.parse_config(config_dict)
        # kwargs allow TRs that need extra ctor args (e.g. q_time_dep for RadiusTR)
        return tr_class(config=cfg, logger=logger, **kwargs)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def eta(self) -> float:
        return self._eta

    # Convenience passthroughs (read from config)
    @property
    def eta_min(self) -> float:
        return float(getattr(self.config, "eta_min", 0.0))

    @property
    def eta_max(self) -> float:
        return float(getattr(self.config, "eta_max", float("inf")))

    @property
    def beta_1(self) -> float:
        return float(getattr(self.config, "beta_1", 0.25))

    @property
    def beta_2(self) -> float:
        return float(getattr(self.config, "beta_2", 0.75))

    @property
    def beta_3(self) -> float:
        return float(getattr(self.config, "beta_3", 0.5))

    @abstractmethod
    def check(self, ctx: TRContext) -> TRCheckResult:
        raise NotImplementedError

    @staticmethod
    def trustworthiness(obj_r: float, obj_r_center: float, obj: float, obj_center: float) -> float:
        delta_obj = obj_center - obj
        delta_obj_r = obj_r_center - obj_r
        return (delta_obj / delta_obj_r) if (delta_obj_r > 0) else np.inf

    def shrink(self) -> None:
        self._eta = max(self._eta * self.beta_3, self.eta_min)

    def enlarge(self) -> None:
        self._eta = min(self._eta / self.beta_3, self.eta_max)

    def eta_too_small(self) -> bool:
        if self.eta <= self.eta_min:
            self.logger.info(
                "Trust region tolerance eta = %3.4e falls below eta_min = %3.4e.",
                self.eta,
                self.eta_min,
            )
            return True
        return False

    def update_by_trustworthiness(
        self,
        obj_r: float,
        obj_r_center: float,
        obj: float,
        obj_center: float,
    ) -> None:
        rho = self.trustworthiness(obj_r, obj_r_center, obj, obj_center)
        if rho > self.beta_2:
            self.enlarge()
            self.logger.info(
                "rho = %3.4e > beta_2 = %3.4e; enlarging eta to %3.4e.",
                rho,
                self.beta_2,
                self.eta,
            )
        else:
            self.logger.info(
                "rho = %3.4e <= beta_2 = %3.4e; keeping eta at %3.4e.",
                rho,
                self.beta_2,
                self.eta,
            )


# ----------------------------
# NoneTR
# ----------------------------

class NoneTR(TR, tr_type=TRType.NONE):
    requires_objective_error = False

    def check(self, ctx: TRContext) -> TRCheckResult:
        return TRCheckResult(True, False)

    def shrink(self) -> None:
        return

    def enlarge(self) -> None:
        return


# ----------------------------
# RadiusTR
# ----------------------------

class RadiusTR(TR, tr_type=TRType.RADIUS):
    requires_objective_error = False

    def __init__(self, config: TrustRegionConfig, q_time_dep: bool, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, logger=logger)
        self.q_time_dep = bool(q_time_dep)

    def check(self, ctx: TRContext) -> TRCheckResult:
        if ctx.center_q is None or ctx.current_q is None or ctx.product is None:
            raise ValueError("RadiusTR requires center_q/current_q/product in TRContext")

        if self.q_time_dep:
            if not isinstance(ctx.product, BochnerProductOperator):
                raise TypeError("For q_time_dep=True, product must be a BochnerProductOperator")

        delta_q = ctx.center_q - ctx.current_q
        norm_delta_q = float(np.sqrt(ctx.product.apply2(delta_q, delta_q)))

        tr_ok = norm_delta_q <= self.eta
        model_insufficient = norm_delta_q > (self.beta_1 * self.eta)
        return TRCheckResult(tr_ok, model_insufficient)


# ----------------------------
# RelativeObjectiveErrorTR
# ----------------------------

class RelativeObjectiveErrorTR(TR, tr_type=TRType.RELATIVE_OBJECTIVE_ERROR):
    requires_objective_error = True

    def __init__(self, config: TrustRegionConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, logger=logger)

    def check(self, ctx: TRContext) -> TRCheckResult:
        J = OBJ.sanitize_objective(ctx.objective, name="objective")
        err = OBJ.sanitize_error(ctx.abs_error, name="abs_error")

        if np.isnan(err):
            raise RuntimeError("RelativeObjectiveErrorTR requires objective error, but got NaN.")

        rel_err = OBJ.rel_error(abs_error=err, objective=J)
        tr_ok = rel_err <= self.eta
        model_insufficient = rel_err > (self.beta_1 * self.eta)
        return TRCheckResult(tr_ok, model_insufficient)
