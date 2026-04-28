from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Optional, Iterable

from RBInvParam.schemas.optimizer import FOMOptimizerCfg, TROptimizerCfg, ArmijoConfig


# ----------------------------
# formatting helpers
# ----------------------------

def _val(x: Any) -> Any:
    """Return enum.value if present, otherwise x."""
    return getattr(x, "value", x)


def _fmt(x: Any) -> str:
    """Pretty formatting for logging."""
    x = _val(x)
    if isinstance(x, float):
        return f"{x:3.4e}"
    return str(x)


def _iter_items(obj: Any) -> Iterable[tuple[str, Any]]:
    """
    Iterate over items for dataclasses, mappings, or objects with __dict__.
    Keeps ordering stable for dataclasses.
    """
    if obj is None:
        return []
    if is_dataclass(obj):
        return asdict(obj).items()
    if isinstance(obj, Mapping):
        return obj.items()
    if hasattr(obj, "__dict__"):
        return vars(obj).items()
    return []


def _log_kv(logger: logging.Logger, key: str, value: Any, indent: str) -> None:
    logger.debug("%s%s : %s", indent, key, _fmt(value))


def _log_block(
    logger: logging.Logger,
    title: str,
    obj: Any,
    *,
    indent: str = "        ",
    header_indent: str = "  ",
    empty_as_none: bool = True,
) -> None:
    """
    Generic block logger. Works for:
      - dict/mapping
      - dataclasses
      - objects with __dict__

    For empty mappings, prints 'None' if empty_as_none=True.
    """
    logger.debug("%s%s :", header_indent, title)

    items = list(_iter_items(obj))
    if not items and empty_as_none:
        logger.debug("%sNone", indent)
        return

    for k, v in items:
        _log_kv(logger, str(k), v, indent)


def _log_armijo(logger: logging.Logger, title: str, cfg: ArmijoConfig, *, indent: str = "  ") -> None:
    logger.debug(
        "%s%s : max_iter=%d initial_step_size=%s kappa_arm=%s shrink=%s",
        indent,
        title,
        cfg.max_iter,
        _fmt(cfg.initial_step_size),
        _fmt(cfg.kappa_arm),
        _fmt(cfg.shrink),
    )


# ----------------------------
# public log functions
# ----------------------------

def log_fom_opt_config(
    logger: logging.Logger,
    cfg: FOMOptimizerCfg,
    *,
    J: float,
    norm_nabla_J: float,
) -> None:
    logger.debug("Running FOM-IRGNM:")
    logger.debug("  J : %s", _fmt(J))
    logger.debug("  norm_nabla_J : %s", _fmt(norm_nabla_J))
    logger.debug("  ")

    # scalar fields
    scalar_fields = [
        ("alpha_0", cfg.alpha_0),
        ("tol", cfg.tol),
        ("tau", cfg.tau),
        ("noise_level", cfg.noise_level),
        ("theta", cfg.theta),
        ("Theta", cfg.Theta),
        ("i_max", cfg.i_max),
        ("reg_loop_max", cfg.reg_loop_max),
        ("i_max_inner", cfg.i_max_inner),
        ("dump_every_nth_loop", cfg.dump_every_nth_loop),
        ("use_cached_operators", cfg.use_cached_operators),
    ]
    for k, v in scalar_fields:
        logger.debug("  %s : %s", k, _fmt(v))

    # dict blocks
    _log_block(logger, "lin_solver_parms", cfg.lin_solver_parms)



def log_tr_opt_config(
    logger: logging.Logger,
    opt_cfg: TROptimizerCfg,
    *,
    J: float,
    norm_nabla_J: float,
) -> None:
    logger.debug("Running Qr-Vr-IRGNM:")
    logger.debug("  J : %s", _fmt(J))
    logger.debug("  norm_nabla_J : %s", _fmt(norm_nabla_J))

    # -------------------------------------------------
    # Scalar fields (auto-detected from dataclass)
    # -------------------------------------------------
    scalar_exclude = {
        "AGC_armijo_cfg",
        "TR_armijo_cfg",
        "TR",
        "lin_solver_parms",
        "enrichment",
        "logging",
        "reductor",
    }

    for name, value in vars(opt_cfg).items():
        if name in scalar_exclude:
            continue
        logger.debug("  %s : %s", name, _fmt(value))

    # -------------------------------------------------
    # Armijo configs
    # -------------------------------------------------
    _log_armijo(logger, "AGC_armijo_cfg", opt_cfg.AGC_armijo_cfg)
    _log_armijo(logger, "TR_armijo_cfg", opt_cfg.TR_armijo_cfg)

    # -------------------------------------------------
    # TR block
    # -------------------------------------------------
    logger.debug("  TR.type : %s", _fmt(opt_cfg.TR.type))

    for name in vars(opt_cfg.TR.config):
        logger.debug(
            "  TR.%s : %s",
            name,
            _fmt(getattr(opt_cfg.TR, name)),  # passthrough
        )

    # -------------------------------------------------
    # Dict blocks
    # -------------------------------------------------
    _log_block(logger, "inner_loop_model_schedule", opt_cfg.inner_loop_model_schedule)
    _log_block(logger, "lin_solver_parms", opt_cfg.lin_solver_parms)
    _log_block(logger, "enrichment", opt_cfg.enrichment)
    _log_block(logger, "logging", opt_cfg.logging)

    # -------------------------------------------------
    # Reductor block (structured)
    # -------------------------------------------------
    logger.debug("  reductor:")
    for name, value in vars(opt_cfg.reductor).items():
        logger.debug("    %s : %s", name, _fmt(value))

    # Optional nested logging.errors shortcut
    if isinstance(opt_cfg.logging, Mapping) and "errors" in opt_cfg.logging:
        logger.debug(
            "  logging.errors : %s",
            _fmt(opt_cfg.logging["errors"]),
        )