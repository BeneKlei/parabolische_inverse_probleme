from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from RBInvParam.optimizer.optimizer_schema import FOMOptimizerCfg, TROptimizerCfg

##########################
# utils
##########################


def _val(x: Any) -> Any:
    """Return enum.value if present, otherwise x."""
    return getattr(x, "value", x)


def _log_mapping(logger: logging.Logger, title: str, m: Optional[Mapping[str, Any]], indent: str = "        ") -> None:
    logger.debug("  %s :", title)
    if not m:
        logger.debug("%sNone", indent)
        return
    for k, v in m.items():
        logger.debug("%s%s : %s", indent, k, _val(v))

##########################
# log funcs
##########################

def log_fom_opt_config(
    logger: logging.Logger,
    cfg: FOMOptimizerCfg,
    *,
    J: float,
    norm_nabla_J: float,
) -> None:
    logger.debug("Running FOM-IRGNM:")
    logger.debug("  J : %3.4e", J)
    logger.debug("  norm_nabla_J : %3.4e", norm_nabla_J)

    logger.debug("  ")
    logger.debug("  alpha_0 : %3.4e", cfg.alpha_0)
    logger.debug("  tol : %3.4e", cfg.tol)
    logger.debug("  tau : %3.4e", cfg.tau)
    logger.debug("  noise_level : %3.4e", cfg.noise_level)
    logger.debug("  theta : %3.4e", cfg.theta)
    logger.debug("  Theta : %3.4e", cfg.Theta)
    logger.debug("  ")
    logger.debug("  i_max : %d", cfg.i_max)
    logger.debug("  reg_loop_max : %d", cfg.reg_loop_max)
    logger.debug("  i_max_inner : %d", cfg.i_max_inner)
    logger.debug("  dump_every_nth_loop : %d", cfg.dump_every_nth_loop)
    logger.debug("  use_cached_operators : %s", cfg.use_cached_operators)

    logger.debug("  lin_solver_parms :")
    for k, v in cfg.lin_solver_parms.items():
        logger.debug("        %s : %s", k, v)

def log_tr_opt_config(
    logger: logging.Logger,
    opt_cfg: TROptimizerCfg,
    *,
    J: float,
    norm_nabla_J: float,
    AGC_armijo_cfg: ArmijoConfig
) -> None:
    logger.debug("Running Qr-Vr-IRGNM:")
    logger.debug("  J : %3.4e", J)
    logger.debug("  norm_nabla_J : %3.4e", norm_nabla_J)

    logger.debug("  alpha_0 : %3.4e", opt_cfg.alpha_0)
    logger.debug("  tol : %3.4e", opt_cfg.tol)
    logger.debug("  tau : %3.4e", opt_cfg.tau)
    logger.debug("  noise_level : %3.4e", opt_cfg.noise_level)
    logger.debug("  theta : %3.4e", opt_cfg.theta)
    logger.debug("  Theta : %3.4e", opt_cfg.Theta)
    logger.debug("  tau_tilde : %3.4e", opt_cfg.tau_tilde)

    logger.debug("  i_max : %d", opt_cfg.i_max)
    logger.debug("  i_max_inner : %d", opt_cfg.i_max_inner)
    logger.debug("  reg_loop_max : %d", opt_cfg.reg_loop_max)

    logger.debug("  TR_enforcement : %s", opt_cfg.TR_enforcement)
    logger.debug("  reg_AGC_step : %s", opt_cfg.reg_AGC_step)
    logger.debug("  use_error_estimator : %s", opt_cfg.use_error_estimator)
    logger.debug("  use_adjoint_space : %s", opt_cfg.use_adjoint_space)
    logger.debug("  offline_parallel : %s", opt_cfg.offline_parallel)
    logger.debug("  use_cached_operators : %s", opt_cfg.use_cached_operators)

    # Armijo configs (both are in the TROptimizerCfg)
    logger.debug(
        "  AGC_armijo_cfg : max_iter=%d initial_step_size=%3.4e kappa_arm=%3.4e shrink=%3.4e",
        AGC_armijo_cfg.max_iter,
        AGC_armijo_cfg.initial_step_size,
        AGC_armijo_cfg.kappa_arm,
        AGC_armijo_cfg.shrink,
    )
    logger.debug(
        "  TR_armijo_cfg  : max_iter=%d initial_step_size=%3.4e kappa_arm=%3.4e shrink=%3.4e",
        opt_cfg.TR_armijo_cfg.max_iter,
        opt_cfg.TR_armijo_cfg.initial_step_size,
        opt_cfg.TR_armijo_cfg.kappa_arm,
        opt_cfg.TR_armijo_cfg.shrink,
    )

    # TR block
    logger.debug("  TR.type : %s", _val(opt_cfg.TR.type))
    logger.debug("  TR.eta_initial : %3.4e", opt_cfg.TR.eta_initial)
    logger.debug("  TR.eta_min : %3.4e", opt_cfg.TR.eta_min)
    logger.debug("  TR.eta_max : %3.4e", opt_cfg.TR.eta_max)
    logger.debug("  TR.beta_1 : %3.4e", opt_cfg.TR.beta_1)
    logger.debug("  TR.beta_2 : %3.4e", opt_cfg.TR.beta_2)
    logger.debug("  TR.beta_3 : %3.4e", opt_cfg.TR.beta_3)

    # dict blocks
    _log_mapping(logger, "lin_solver_parms", opt_cfg.lin_solver_parms)
    _log_mapping(logger, "enrichment", opt_cfg.enrichment)
    _log_mapping(logger, "error_estimator_types", opt_cfg.error_estimator_types)
    _log_mapping(logger, "logging", opt_cfg.logging)

    # Commonly used nested keys (optional, but handy)
    if isinstance(opt_cfg.logging, dict):
        if "errors" in opt_cfg.logging:
            logger.debug("  logging.errors : %s", _val(opt_cfg.logging.get("errors")))