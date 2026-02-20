from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

from RBInvParam.trust_region import TRType
from RBInvParam.schemas.trust_region import TrustRegionConfig
from RBInvParam.schemas.reductor import InstationaryReductorConfig
from RBInvParam.schemas.utils import unknown_keys, require


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

        unknown_keys(data, cls.__dataclass_fields__.keys(), where=where)

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
    """Combine TR 'type' + TrustRegionConfig; forward attribute access to config."""
    type: TRType
    config: TrustRegionConfig

    def validate(self) -> None:
        self.config.validate()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.config, name)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, where: str) -> "TRBlock":
        if not isinstance(data, dict):
            raise TypeError(f"{where} must be a dict")

        if "type" not in data:
            raise KeyError(f"Missing keys in {where}: ['type']")

        block = dict(data)  # copy
        type_raw = block.pop("type")

        tr_type = TRType(type_raw) if isinstance(type_raw, str) else type_raw
        if not isinstance(tr_type, TRType):
            raise TypeError(f"{where}['type'] must be TRType or str convertible to TRType")

        # remaining keys belong to TrustRegionConfig
        tr_cfg = TrustRegionConfig.from_dict(block)
        tr = cls(type=tr_type, config=tr_cfg)
        tr.validate()
        return tr

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

    # ----------------------------
    # Construction
    # ----------------------------

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        where: str = "optimizer_parameter",
    ) -> "FOMOptimizerCfg":

        allowed = set(cls.__dataclass_fields__.keys())
        unknown_keys(data, allowed, where=where)

        require(
            data,
            [
                "method", "q_0",
                "alpha_0", "tol", "tau", "noise_level",
                "theta", "Theta",
                "i_max", "reg_loop_max", "i_max_inner",
                "use_cached_operators", "dump_every_nth_loop",
                "lin_solver_parms",
            ],
            where=where,
        )

        cfg = cls(
            method=str(data["method"]),
            q_0=data["q_0"].copy() if hasattr(data["q_0"], "copy") else data["q_0"],

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

    # ----------------------------
    # Validation
    # ----------------------------

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

        if not isinstance(self.lin_solver_parms, dict):
            raise ValueError("lin_solver_parms must be a dict")

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
    reg_AGC_step: bool
    TR_enforcement: str
    dump_every_nth_loop: int

    reductor: InstationaryReductorConfig

    lin_solver_parms: Dict[str, Any]
    enrichment: Dict[str, Any]
    logging: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, where: str = "optimizer_parameter") -> "TROptimizerCfg":
        allowed = set(cls.__dataclass_fields__.keys())
        unknown_keys(data, allowed, where=where)

        require(
            data,
            [
                "method", "q_0", "alpha_0", "tol", "tau", "noise_level", "theta", "Theta", "tau_tilde",
                "i_max", "reg_loop_max", "i_max_inner",
                "AGC_armijo_cfg", "TR_armijo_cfg", "TR",
                "use_cached_operators", "use_error_estimator",
                "reg_AGC_step", "TR_enforcement", "dump_every_nth_loop",
                "reductor",
                "lin_solver_parms", "enrichment",
                "logging",
            ],
            where=where,
        )

        agc = ArmijoConfig.from_dict(data["AGC_armijo_cfg"], where=f"{where}['AGC_armijo_cfg']")
        tr_arm = ArmijoConfig.from_dict(data["TR_armijo_cfg"], where=f"{where}['TR_armijo_cfg']")
        tr = TRBlock.from_dict(data["TR"], where=f"{where}['TR']")
        red = InstationaryReductorConfig.from_dict(data["reductor"], where=f"{where}['reductor']")

        cfg = cls(
            method=str(data["method"]),
            q_0=data["q_0"].copy() if hasattr(data["q_0"], "copy") else data["q_0"],
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
            reg_AGC_step=bool(data["reg_AGC_step"]),
            TR_enforcement=str(data["TR_enforcement"]),
            dump_every_nth_loop=int(data["dump_every_nth_loop"]),

            reductor=red,

            lin_solver_parms=dict(data["lin_solver_parms"]),
            enrichment=dict(data["enrichment"]),
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

        self.AGC_armijo_cfg.validate()
        self.TR_armijo_cfg.validate()
        self.TR.validate()
        self.reductor.validate()

        if "errors" not in self.logging:
            raise ValueError("logging must contain key 'errors'")