from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union


class TCCPerturbationType(str, Enum):
    GRADIENT_DIRECTION = "gradient_direction"
    LOCAL_BASIS = "local_basis"
    Q_EXACT_DIRECTION = "q_exact_direction"


@dataclass(frozen=True)
class TCCEvaluatorConfig:
    amplitudes: tuple[float, ...]
    max_h: int
    seed: int
    perturbation_mode: TCCPerturbationType
    create_pdf: bool
    pdf_filename: Union[str, Path]
    verbose_logging: bool

    @classmethod
    def defaults(cls) -> "TCCEvaluatorConfig":
        return cls(
            amplitudes=(1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10),
            max_h=1,
            seed=0,
            perturbation_mode=TCCPerturbationType.Q_EXACT_DIRECTION,
            create_pdf=False,
            pdf_filename="tcc_analysis.pdf",
            verbose_logging=False
        )

    def validate(self) -> None:
        if len(self.amplitudes) == 0:
            raise ValueError("amplitudes must not be empty")
        if any(a < 0 for a in self.amplitudes):
            raise ValueError("all amplitudes must be >= 0")
        if self.max_h < 1:
            raise ValueError("max_h must be >= 1")
        if not isinstance(self.perturbation_mode, TCCPerturbationType):
            raise ValueError("perturbation_mode must be of type TCCPerturbationType")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TCCEvaluatorConfig":
        base = cls.defaults()
        if not data:
            base.validate()
            return base

        valid_keys = set(cls.__dataclass_fields__.keys())
        unknown = set(data.keys()) - valid_keys
        if unknown:
            raise ValueError(f"Unknown TCCEvaluatorConfig keys: {sorted(unknown)}")

        mode = data.get("perturbation_mode", base.perturbation_mode)
        if isinstance(mode, str):
            mode = TCCPerturbationType(mode)

        cfg = replace(
            base,
            amplitudes=tuple(data.get("amplitudes", base.amplitudes)),
            max_h=int(data.get("max_h", base.max_h)),
            seed=int(data.get("seed", base.seed)),
            perturbation_mode=mode,
            create_pdf=bool(data.get("create_pdf", base.create_pdf)),
            pdf_filename=data.get("pdf_filename", base.pdf_filename),
            verbose_logging=bool(data.get("verbose_logging", base.verbose_logging)),
        )
        cfg.validate()
        return cfg