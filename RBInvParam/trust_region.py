from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

from RBInvParam.utils.logger import get_default_logger


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class TrustRegionConfig:
    eta: float
    eta_min: float
    eta_max: float
    beta_1: float
    beta_2: float
    beta_3: float

    @classmethod
    def defaults(cls) -> "TrustRegionConfig":
        return cls(
            eta=1.0,
            eta_min=0.0,
            eta_max=float("inf"),
            beta_1=0.25,
            beta_2=0.75,
            beta_3=0.5,
        )

    def validate(self) -> None:
        if self.eta <= 0:
            raise ValueError("eta must be > 0")
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
            eta=float(data.get("eta", base.eta)),
            eta_min=float(data.get("eta_min", base.eta_min)),
            eta_max=float(data.get("eta_max", base.eta_max)),
            beta_1=float(data.get("beta_1", base.beta_1)),
            beta_2=float(data.get("beta_2", base.beta_2)),
            beta_3=float(data.get("beta_3", base.beta_3)),
        )
        cfg.validate()
        return cfg


# ----------------------------
# Enum
# ----------------------------

class TRType(str, Enum):
    NONE = "none"
    RELATIVE_OBJECTIVE_ERROR = "relative_objective_error"


# ----------------------------
# Base class
# ----------------------------

class TR(ABC):
    """
    Base trust-region controller.

    Design:
      - self.config is immutable (frozen dataclass)
      - self._eta is the only mutable state
      - all constants are accessed via self.config
    """
    _registry: Dict[TRType, type["TR"]] = {}

    def __init_subclass__(cls, *, tr_type: Optional[TRType] = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if tr_type is not None:
            TR._registry[tr_type] = cls

    def __init__(self, config: TrustRegionConfig, logger: Optional[logging.Logger] = None):
        config.validate()
        self.config = config
        self._eta = config.eta

        self._logger = logger or get_default_logger(self.__class__.__name__)
        self._logger.setLevel(logging.DEBUG)
        self._logger.debug("Setting up %s", self.__class__.__name__)

    @classmethod
    def from_type(
        cls,
        tr_type: TRType,
        config_dict: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> "TR":
        config = TrustRegionConfig.from_dict(config_dict)

        try:
            tr_class = cls._registry[tr_type]
        except KeyError:
            raise ValueError(f"No TR registered for type {tr_type}")

        return tr_class(config=config, logger=logger)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def eta(self) -> float:
        return self._eta

    @eta.setter
    def eta(self, value: float) -> None:
        self._eta = float(value)

    # Convenience passthroughs (no "unpacking"; read from config)
    @property
    def eta_min(self) -> float:
        return self.config.eta_min

    @property
    def eta_max(self) -> float:
        return self.config.eta_max

    @property
    def beta_1(self) -> float:
        return self.config.beta_1

    @property
    def beta_2(self) -> float:
        return self.config.beta_2

    @property
    def beta_3(self) -> float:
        return self.config.beta_3

    @abstractmethod
    def check(self, **kwargs) -> bool:
        """Return True if the current trust-region criterion is satisfied."""
        raise NotImplementedError

    @staticmethod
    def trustworthiness(obj_r: float, obj_r_center: float, obj: float, obj_center: float) -> float:
        delta_obj = obj_center - obj
        delta_obj_r = obj_r_center - obj_r
        return (delta_obj / delta_obj_r) if (delta_obj_r > 0) else np.inf

    def shrink(self) -> None:
        self.eta = max(self.eta * self.beta_3, self.eta_min)

    def enlarge(self) -> None:
        self.eta = min(self.eta / self.beta_3, self.eta_max)

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
    def check(self, **kwargs) -> bool:
        return True

    def shrink(self) -> None:
        return

    def enlarge(self) -> None:
        return


# ----------------------------
# RelativeObjectiveErrorTR
# ----------------------------

class RelativeObjectiveErrorTR(TR, tr_type=TRType.RELATIVE_OBJECTIVE_ERROR):
    def __init__(self, config: TrustRegionConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, logger=logger)

    def check(self, *, objective: float, abs_error: float) -> bool:
        assert objective >= 0
        if objective == 0:
            return False
        return (abs_error / objective) <= self.eta

