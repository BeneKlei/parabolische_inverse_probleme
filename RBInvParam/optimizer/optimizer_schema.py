from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Mapping, Iterable

from RBInvParam.trust_region import TRType, TrustRegionConfig


# ----------------------------
# helpers
# ----------------------------

def _unknown_keys(data: Mapping[str, Any], allowed: Iterable[str], *, where: str) -> None:
    unknown = set(data.keys()) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {where}: {sorted(unknown)}")


def _require(data: Mapping[str, Any], keys: Iterable[str], *, where: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise KeyError(f"Missing keys in {where}: {missing}")


# ----------------------------
# Armijo config
# ----------------------------

@dataclass(frozen=True)
class ArmijoConfig:
    max_iter: int
    initial_step_size: float
    kappa_arm: float
    shrink: float = 0.5

    @classmethod
    def defaults(cls) -> "ArmijoConfig":
        return cls(max_iter=50, initial_step_size=1.0, kappa_arm=1e-12, shrink=0.5)

    def validate(self) -> None:
        if self.max_iter <= 0:
            raise ValueError("ArmijoConfig.max_iter must be > 0")
        if self.initial_step_size <= 0:
            raise ValueError("ArmijoConfig.initial_step_size must be > 0")
        if self.kappa_arm <= 0:
            raise ValueError("ArmijoConfig.kappa_arm must be > 0")
        if not (0.0 < self.shrink < 1.0):
            raise ValueError("ArmijoConfig.shrink must be in (0, 1)")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]], *, where: str) -> "ArmijoConfig":
        base = cls.defaults()
        if not data:
            base.validate()
            return base

        _unknown_keys(data, cls.__dataclass_fields__.keys(), where=where)

        cfg = replace(
            base,
            max_iter=int(data.get("max_iter", base.max_iter)),
            initial_step_size=float(data.get("initial_step_size", base.initial_step_size)),
            kappa_arm=float(data.get("kappa_arm", base.kappa_arm)),
            shrink=float(data.get("shrink", base.shrink)),
        )
        cfg.validate()
        return cfg


# ----------------------------
# TR block wrapper (Option B)
# ----------------------------

@dataclass(frozen=True)
class TRBlock:
    """
    Combines TR 'type' + the actual TrustRegionConfig so callers can do:

        opt_cfg.TR.type
        opt_cfg.TR.eta_initial   (via passthrough)
        opt_cfg.TR.config        (explicit TrustRegionConfig)

    """
    type: TRType
    config: TrustRegionConfig

    def validate(self) -> None:
        self.config.validate()

    # Convenience passthrough so you can keep writing opt_cfg.TR.eta_initial etc.
    def __getattr__(self, name: str) -> Any:
        return getattr(self.config, name)


# ----------------------------
# Normal run schema
# ----------------------------

@dataclass(frozen=True)
class FOMOptimizerCfg:
    method: str
    q_0: Any
    alpha_0: float
    tol: float
    tau: float
    noise_level: float
    theta: float
    Theta: float

    i_max: int
    reg_loop_max: int
    i_max_inner: int

    use_cached_operators: bool
    dump_every_nth_loop: int

    lin_solver_parms: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, where: str = "optimizer_parameter") -> "FOMOptimizerCfg":
        allowed = set(cls.__dataclass_fields__.keys())
        _unknown_keys(data, allowed, where=where)

        _require(
            data,
            [
                "method", "q_0", "alpha_0", "tol", "tau", "noise_level", "theta", "Theta",
                "i_max", "reg_loop_max", "i_max_inner",
                "use_cached_operators", "dump_every_nth_loop",
                "lin_solver_parms",
            ],
            where=where
        )

        cfg = cls(
            method = str(data["method"]),
            q_0=data["q_0"].copy(),
            alpha_0=float(data["alpha_0"]),
            tol=float(data["tol"]),
            tau=float(data["tau"]),
            noise_level=float(data["noise_level"]),
            theta=float(data["theta"]),
            Theta=float(data["Theta"]),

            i_max=int(data["i_max"]),
            reg_loop_max=int(data["reg_loop_max"]),
            i_max_inner=int(data["i_max_inner"]),

            use_cached_operators=bool(data["use_cached_operators"]),
            dump_every_nth_loop=int(data["dump_every_nth_loop"]),

            lin_solver_parms=dict(data["lin_solver_parms"]),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.alpha_0 < 0:
            raise ValueError("alpha_0 must be >= 0")
        if self.tol <= 0:
            raise ValueError("tol must be > 0")
        if self.tau <= 0:
            raise ValueError("tau must be > 0")
        if self.noise_level < 0:
            raise ValueError("noise_level must be >= 0")
        if not (0 < self.theta < self.Theta):
            raise ValueError("Require 0 < theta < Theta")
        if self.i_max < 1 or self.i_max_inner < 1 or self.reg_loop_max < 1:
            raise ValueError("i_max, i_max_inner, reg_loop_max must be >= 1")


# ----------------------------
# TR run schema
# ----------------------------

@dataclass(frozen=True)
class TROptimizerCfg:
    method: str
    q_0: Any
    alpha_0: float
    tol: float
    tau: float
    noise_level: float
    theta: float
    Theta: float
    tau_tilde: float

    i_max: int
    reg_loop_max: int
    i_max_inner: int

    AGC_armijo_cfg: ArmijoConfig
    TR_armijo_cfg: ArmijoConfig

    TR: TRBlock

    use_cached_operators: bool
    use_error_estimator: bool
    use_adjoint_space: bool
    offline_parallel: bool
    reg_AGC_step: bool
    TR_enforcement: str
    dump_every_nth_loop: int

    lin_solver_parms: Dict[str, Any]
    enrichment: Dict[str, Any]

    error_estimator_types: Dict[str, Any]
    logging: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, where: str = "optimizer_parameter") -> "TROptimizerCfg":
        allowed = set(cls.__dataclass_fields__.keys())
        _unknown_keys(data, allowed, where=where)

        _require(
            data,
            [
                "method", "q_0", "alpha_0", "tol", "tau", "noise_level", "theta", "Theta", "tau_tilde",
                "i_max", "reg_loop_max", "i_max_inner",
                "AGC_armijo_cfg", "TR_armijo_cfg", "TR",
                "use_cached_operators", "use_error_estimator", "use_adjoint_space", "offline_parallel",
                "reg_AGC_step", "TR_enforcement", "dump_every_nth_loop",
                "lin_solver_parms", "enrichment",
                "error_estimator_types", "logging",
            ],
            where=where
        )

        agc = ArmijoConfig.from_dict(data["AGC_armijo_cfg"], where=f"{where}['AGC_armijo_cfg']")
        tr_arm = ArmijoConfig.from_dict(data["TR_armijo_cfg"], where=f"{where}['TR_armijo_cfg']")

        # --- TR block: keep "type" and config together ---
        tr_block = dict(data["TR"])

        if "type" not in tr_block:
            raise KeyError(f"Missing keys in {where}['TR']: ['type']")

        tr_type_raw = tr_block.pop("type")
        tr_type = TRType(tr_type_raw) if isinstance(tr_type_raw, str) else tr_type_raw

        tr_cfg = TrustRegionConfig.from_dict(tr_block)
        tr = TRBlock(type=tr_type, config=tr_cfg)

        cfg = cls(
            method = str(data["method"]),
            q_0=data["q_0"].copy(),
            alpha_0=float(data["alpha_0"]),
            tol=float(data["tol"]),
            tau=float(data["tau"]),
            noise_level=float(data["noise_level"]),
            theta=float(data["theta"]),
            Theta=float(data["Theta"]),
            tau_tilde=float(data["tau_tilde"]),

            i_max=int(data["i_max"]),
            reg_loop_max=int(data["reg_loop_max"]),
            i_max_inner=int(data["i_max_inner"]),

            AGC_armijo_cfg=agc,
            TR_armijo_cfg=tr_arm,

            TR=tr,

            use_cached_operators=bool(data["use_cached_operators"]),
            use_error_estimator=bool(data["use_error_estimator"]),
            use_adjoint_space=bool(data["use_adjoint_space"]),
            offline_parallel=bool(data["offline_parallel"]),
            reg_AGC_step=bool(data["reg_AGC_step"]),
            TR_enforcement=str(data["TR_enforcement"]),
            dump_every_nth_loop=int(data["dump_every_nth_loop"]),

            lin_solver_parms=dict(data["lin_solver_parms"]),
            enrichment=dict(data["enrichment"]),

            error_estimator_types=dict(data["error_estimator_types"]),
            logging=dict(data["logging"]),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.alpha_0 < 0:
            raise ValueError("alpha_0 must be >= 0")
        if self.tol <= 0:
            raise ValueError("tol must be > 0")
        if self.tau <= 0:
            raise ValueError("tau must be > 0")
        if self.noise_level < 0:
            raise ValueError("noise_level must be >= 0")
        if not (0 < self.theta < self.Theta):
            raise ValueError("Require 0 < theta < Theta")
        if self.tau_tilde <= 0:
            raise ValueError("tau_tilde must be > 0")

        if self.i_max < 1 or self.i_max_inner < 1 or self.reg_loop_max < 1:
            raise ValueError("i_max, i_max_inner, reg_loop_max must be >= 1")

        if self.TR_enforcement not in ("check_error", "backtracking"):
            raise ValueError("TR_enforcement must be 'check_error' or 'backtracking'")

        # nested validate
        self.AGC_armijo_cfg.validate()
        self.TR_armijo_cfg.validate()
        self.TR.validate()

        # light sanity checks for the pass-through dicts
        if "errors" not in self.logging:
            raise ValueError("logging must contain key 'errors'")
        if not isinstance(self.error_estimator_types, dict):
            raise ValueError("error_estimator_types must be a dict")
