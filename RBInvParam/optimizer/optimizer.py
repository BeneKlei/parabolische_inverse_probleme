import logging
import numpy as np
import sys

from abc import abstractmethod
from typing import Dict, Union, Tuple, List, Optional, Any
from timeit import default_timer as timer
from pathlib import Path
from enum import Enum
from dataclasses import replace

import pymor_dealii_bindings as pd2

from pymor.vectorarrays.interface import VectorArray
from pymor.algorithms.hapod import inc_vectorarray_hapod
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace
from pymor.operators.interface import Operator
from pymor.core.base import BasicObject
from pymor.core.exceptions import ExtensionError

from RBInvParam.model import InstationaryModelIP
from RBInvParam.linear_solver.gradient_descent import gradient_descent_linearized_problem
from RBInvParam.linear_solver.BiCGSTAB import BiCGStab_linearized_problem
from RBInvParam.snapshot_preprocessor import SnapshotPreprocessor
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.utils.io import save_dict_to_pkl, dealii_vector_space_to_numpy
from RBInvParam.domain_projector import SimpleBoundDomainProjector, ProjectionMismatchError
from RBInvParam.trust_region import *

from RBInvParam.schemas.optimizer import FOMOptimizerCfg, TROptimizerCfg, ArmijoConfig, ModelScheduleBlock
from RBInvParam.optimizer.error_evaluator import ErrorEvaluator 
from RBInvParam.optimizer.numerics import GLOBAL_OBJ_POLICY as OBJ
from RBInvParam.schemas.logging_optimizer import log_fom_opt_config, log_tr_opt_config

from RBInvParam.reduction.build import build_reductor
from RBInvParam.schemas.reductor import InstationaryReductorConfig, LinearizationMethod

from RBInvParam.schemas.tcc_evaluator import TCCEvaluatorConfig
from RBInvParam.optimizer.tcc_evaluator import TCCEvaluator


MACHINE_EPS = sys.float_info.epsilon
STAGNATION_TOL = 1e-6

#######################################################################

class LoggerErrorChoice(Enum):
    NONE = "none"
    ALL = "all"
    OBJECTIVE = "objective"
    GRADIENT = "gradient"


#######################################################################

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
fig_1, ax_1 = plt.subplots(figsize=(6,4))
fig_2, ax_2 = plt.subplots(figsize=(6,4))
fig_3, ax_3 = plt.subplots(figsize=(6,4))

#cmap = plt.cm.get_cmap('viridis', 30) 
cmap = plt.cm.get_cmap('tab10')
colors = cmap.colors * 5   # repeat 5 times
cmap = ListedColormap(colors)

#######################################################################

class Optimizer(BasicObject):
    def __init__(self, 
                 optimizer_parameter: Dict, 
                 FOM : InstationaryModelIP,
                 save_path: Path,
                 logger: logging.Logger = None) -> None:
                 
        self.FOM = FOM
        self.optimizer_parameter = optimizer_parameter

        if self.optimizer_parameter['noise_level'] is None:
            self.optimizer_parameter['noise_level'] = FOM.setup['noise_info']['abs_noise_level_y']

        self._check_optimizer_parameter()    
        logging.basicConfig()

        if logger:
            self._logger = logger
        else:
            self._logger = get_default_logger(self.__class__.__name__)
            self._logger.setLevel(logging.DEBUG)
        self.logger.debug(f"Setting up {self.__class__.__name__}")

        save_path = Path(save_path)
        assert save_path.exists()
        self.save_path = save_path

        self._setup_TR(optimizer_parameter)
        self.error_evaluator = ErrorEvaluator(FOM=self.FOM, logger=self.logger)

        self.name = None
        self.IRGNM_idx = 0
        self.IRGNM_statistics = {}

        self.linear_solver_operator = None
        self.last_update_q = None
 
        self.I = 0
        self.FOM_projector = SimpleBoundDomainProjector(
            model = FOM,
            bounds = FOM.bounds,
            reductor = None,
            use_sufficient_condition = False,        
            logger = self.logger
        )

        self.estimate_tcc = optimizer_parameter["logging"].get("estimate_tcc", None)

        self.tcc_evaluator = None
        if self.estimate_tcc:
            tcc_config = TCCEvaluatorConfig.from_dict(
                self.estimate_tcc.get("config")
            )
            self.tcc_evaluator = TCCEvaluator(
                config=tcc_config,
                logger=self.logger,
            )

        if optimizer_parameter["logging"]["errors"] == LoggerErrorChoice.NONE:
            self.error_estimate_targets_outer = ['J']
            self.error_estimate_targets_inner = []
        elif optimizer_parameter["logging"]["errors"] == LoggerErrorChoice.OBJECTIVE:
            self.error_estimate_targets_outer = ['J']
            self.error_estimate_targets_inner = ['J']
        elif optimizer_parameter["logging"]["errors"] == LoggerErrorChoice.GRADIENT:
            self.error_estimate_targets_outer = ['J', 'nabla_J']
            self.error_estimate_targets_inner = ['J', 'nabla_J']
        elif optimizer_parameter["logging"]["errors"] == LoggerErrorChoice.ALL:
            self.error_estimate_targets_outer = ['J', 'nabla_J']
            self.error_estimate_targets_inner = ['J', 'nabla_J', 'lin_J', 'nabla_lin_J']
        else:
            raise ValueError
        
        self.error_estimate_targets_outer = list(set(self.error_estimate_targets_outer))
        self.error_estimate_targets_inner = list(set(self.error_estimate_targets_inner))

    def _setup_TR(self, optimizer_parameter: Dict[str, Any]) -> None:
        tr_type = TRType.NONE
        tr_cfg = optimizer_parameter.get("TR")

        if not tr_cfg:
            self.TR = TR.from_type(TRType.NONE, {}, logger=self.logger)
        else:
            tr_type = tr_cfg.get("type", TRType.NONE)
            tr_cfg_no_type = {k: v for k, v in tr_cfg.items() if k != "type"}

            # If you use RadiusTR, it needs q_time_dep (and later product in ctx)
            extra_kwargs = {}
            if tr_type == TRType.RADIUS:
                extra_kwargs["q_time_dep"] = bool(getattr(self.FOM, "q_time_dep", False))

            self.TR = TR.from_type(tr_type, tr_cfg_no_type, logger=self.logger, **extra_kwargs)

        self.logger.info(
            "TR configured: type=%s requires_obj_err=%s eta=%g eta_min=%g eta_max=%g beta_1=%g beta_2=%g beta_3=%g",
            getattr(tr_type, "value", str(tr_type)),
            getattr(self.TR, "requires_objective_error", None),
            self.TR.eta, self.TR.eta_min, self.TR.eta_max,
            self.TR.beta_1, self.TR.beta_2, self.TR.beta_3,
        )

    def _check_optimizer_parameter(self) -> None:
        keys = self.optimizer_parameter.keys()

        assert self.optimizer_parameter["alpha_0"] >= 0
        assert self.optimizer_parameter["tol"] > 0
        assert self.optimizer_parameter["tau"] > 0
        assert self.optimizer_parameter["noise_level"] >= 0
        assert 0 < self.optimizer_parameter["theta"] \
                 < self.optimizer_parameter["Theta"] 
                 #\ < 1
        if "tau_tilde" in keys:
            assert self.optimizer_parameter["tau_tilde"] > 0

        assert self.optimizer_parameter["i_max"] >= 1
        if "i_max_inner" in keys:
            assert self.optimizer_parameter["i_max_inner"] >= 1
        assert self.optimizer_parameter["reg_loop_max"] >= 1
        if "armijo_max_iter" in keys:
            assert self.optimizer_parameter["armijo_max_iter"] >= 1

        if "eta0" in keys:
            assert self.optimizer_parameter["eta0"] > 0
        if "kappa_arm" in keys:
            assert self.optimizer_parameter["kappa_arm"] > 0
        if "beta_1" in keys:
            assert 0 < self.optimizer_parameter["beta_1"] < 1
        if "beta_2" in keys:
            assert 3/4 <= self.optimizer_parameter["beta_2"] < 1
        if "beta_3" in keys:
            assert 0 < self.optimizer_parameter["beta_3"] < 1    
    
    def _tr_product(self, model: InstationaryModelIP):
        """
        Product to use for RadiusTR checks. Centralize the choice.
        """
        # adjust names to your actual model products
        if getattr(model, "q_time_dep", False):
            return model.products["bochner_prod_Q"]
        return model.products["prod_Q"]

    def _maybe_estimate_tr_error(
        self,
        *,
        model: InstationaryModelIP,
        previous_q: NumpyVectorArray,
        current_q: NumpyVectorArray,
        u: VectorArray,
        p: VectorArray,
        u_dot: Optional[VectorArray],
        p_dot: Optional[VectorArray],
        current_J: float,
        targets: List[str],
        use_cached_operators: bool,
        use_error_estimator: bool,
    ) -> Dict[str, Any]:
        
        """
        Compute abs objective error ONLY if the current TR needs it AND estimator is enabled.
        Returns (abs_err_J, errors_dict).
        """
        # if not getattr(self.TR, "requires_objective_error", True):
        #     return {}
        
        diff = current_q - previous_q
        d_r = diff if diff.norm().max() > MACHINE_EPS else None

        errors = self.error_evaluator.estimate_errors(
            model=model,
            reductor = getattr(self, "reductor", None),
            q_r=current_q,
            d_r=d_r,
            u_r=u,
            p_r=p,
            u_dot_r=u_dot,
            p_dot_r=p_dot,
            J_r=current_J,
            targets=targets,
            use_cached_operators=use_cached_operators,
            use_error_estimator=use_error_estimator,
        )
    
        return errors

    def _eval_bt_step(
        self,
        model: InstationaryModelIP,
        previous_q: NumpyVectorArray,
        tr_center_q: NumpyVectorArray,
        step_size: float,
        search_direction: NumpyVectorArray,
        projector: Optional[SimpleBoundDomainProjector],
        use_cached_operators: bool,
        alpha: float,
        use_error_estimator: bool,
    ) -> Tuple[NumpyVectorArray, float, Dict[str, Any], TRContext]:

        if projector is not None:
            projector.pre_compute(center=previous_q)
            try:
                current_q: NumpyVectorArray = projector.project_domain(
                    previous_q, step_size * search_direction
                )
            except ProjectionMismatchError:
                current_q = previous_q
        else:
            current_q = previous_q + step_size * search_direction

        u, u_dot = model.solve_state(
            q=current_q,
            use_cached_operators=use_cached_operators,
            return_higher_orders=True,
        )
        p, p_dot = model.solve_adjoint(
            q=current_q,
            u=u,
            use_cached_operators=use_cached_operators,
            return_higher_orders=True,
        )

        current_J: float = model.objective(u=u, q=current_q, alpha=alpha)

        # ---- compute objective error ONLY if required ----
        errors = self._maybe_estimate_tr_error(
            model=model,
            previous_q=previous_q,
            current_q=current_q,
            u=u,
            p=p,
            u_dot=u_dot,
            p_dot=p_dot,
            current_J=current_J,
            targets=self.error_estimate_targets_inner,
            use_cached_operators=use_cached_operators,
            use_error_estimator=use_error_estimator,
        )

        # ---- build TRContext (includes fields needed by RadiusTR too) ----
        ctx = TRContext(
            objective=float(current_J),
            abs_error=float(errors.get("err_J", np.nan)),
            center_q=tr_center_q,
            current_q=current_q,
            product=self._tr_product(model),
            step_size=float(step_size),
            meta=None,
        )
        # Note: product is harmless for non-RadiusTR; RadiusTR will require it.

        return current_q, current_J, errors, ctx

    def _armijo_TR_line_serach(
        self,
        model: InstationaryModelIP,
        previous_q: NumpyVectorArray,
        tr_center_q: NumpyVectorArray,
        previous_J: float,
        search_direction: NumpyVectorArray,
        armijo_cfg: ArmijoConfig,
        use_cached_operators: bool = False,
        projector: Optional[SimpleBoundDomainProjector] = None,
        alpha: float = 0.0,
        use_error_estimator: bool = False,
    ) -> Tuple[NumpyVectorArray, float, bool, bool, float, Dict[str, Any]]:
        
        step_size = armijo_cfg.initial_step_size

        if armijo_cfg.max_iter <= 0:
            raise ValueError("armijo.max_iter must be > 0")
        if step_size <= 0:
            raise ValueError("armijo.initial_step_size must be > 0")
        if not (0.0 < armijo_cfg.shrink < 1.0):
            raise ValueError("armijo.shrink must be in (0,1)")

        self.logger.info("Start Armijo backtracking, with J = %3.4e.", previous_J)
        errors: Dict[str, Any] = {}

        # initialize to keep type-checkers + avoid unbound locals
        errors: Dict[str, Any] = {}
        current_q = previous_q
        current_J = previous_J
        armijo_ok = False
        tr_ok = False
        model_insufficient = False

        i = 0
        while i < armijo_cfg.max_iter:
            current_q, current_J, errors, ctx = self._eval_bt_step(
                model=model,
                previous_q=previous_q,
                tr_center_q=tr_center_q,
                step_size=step_size,
                search_direction=search_direction,
                projector=projector,
                use_cached_operators=use_cached_operators,
                alpha=alpha,
                use_error_estimator=use_error_estimator,
            )

            # Armijo
            norm_d = model.compute_gradient_norm(previous_q - current_q)
            lhs = previous_J - current_J
            rhs = armijo_cfg.kappa_arm / step_size * norm_d**2

            print(lhs)
            print(rhs)

            if abs(lhs) <= MACHINE_EPS:
                lhs = 0.0
            if abs(rhs) <= MACHINE_EPS:
                rhs = 0.0
            armijo_ok = lhs >= rhs

            # Trust region (no error computed unless TR needs it)
            tr_res = self.TR.check(ctx)
            tr_ok = tr_res.tr_ok
            model_insufficient = tr_res.model_insufficient

            if armijo_ok and tr_ok:
                break

            step_size *= armijo_cfg.shrink
            i += 1

        TR_max_iter_cond = (i >= armijo_cfg.max_iter)

        condition = (armijo_ok and tr_ok)
        if not condition:
            self.logger.error(
                "Armijo backtracking did NOT terminate normally. step_size=%3.4e; J=%3.4e",
                step_size, current_J
            )
            self.logger.debug("armijo_ok=%s, tr_ok=%s, eta=%3.4e", armijo_ok, tr_ok, self.TR.eta)
        else:
            self.logger.debug(
                "Armijo backtracking terminated. step_size=%3.4e; J=%3.4e; eta=%3.4e",
                step_size, current_J, self.TR.eta
            )
        
        return current_q, current_J, model_insufficient, TR_max_iter_cond, step_size, errors
    
    def _evaluate_TCC(self, q=None) -> Dict[str, Any]:
        if not self.estimate_tcc or self.tcc_evaluator is None:
            return {}

        model_keys = set(self.estimate_tcc.get("models", []))
        valid_model_keys = {"FOM", "ROM"}

        unknown_keys = model_keys - valid_model_keys
        if unknown_keys:
            raise ValueError(
                f"Unknown TCC model keys: {sorted(unknown_keys)}. "
                f"Allowed keys are: {sorted(valid_model_keys)}"
            )

        models: Dict[str, Any] = {}
        q_map: Dict[str, Any] = {}

        if "FOM" in model_keys:
            models["FOM"] = self.FOM

        if "ROM" in model_keys:
            if getattr(self, "QrVrROM", None) is None:
                raise ValueError(
                    "Requested TCC evaluation for 'ROM', but self.QrVrROM is None."
                )
            models["ROM"] = self.QrVrROM

        if q is not None:
            q_in_fom = q in self.FOM.Q
            q_in_rom = ("ROM" in models) and (q in self.QrVrROM.Q)

            if q_in_fom and q_in_rom:
                raise ValueError("Given q belongs to both FOM and ROM spaces; ambiguous input.")

            if not q_in_fom and not q_in_rom:
                raise ValueError("Given q is neither in self.FOM.Q nor in self.QrVrROM.Q.")

            if q_in_fom:
                if "FOM" in models:
                    q_map["FOM"] = q
                if "ROM" in models:
                    q_map["ROM"] = self.reductor.project_vectorarray(q, "parameter_basis")

            elif q_in_rom:
                if "ROM" in models:
                    q_map["ROM"] = q
                if "FOM" in models:
                    q_map["FOM"] = self.reductor.reconstruct(q, basis="parameter_basis")

        else:
            last_update_q = getattr(self, "last_update_q", None)

            if last_update_q is not None:
                if "ROM" in models:
                    try:
                        if last_update_q in self.QrVrROM.Q:
                            q_map["ROM"] = last_update_q
                    except Exception:
                        pass

                if "FOM" in models:
                    try:
                        if "ROM" in q_map:
                            q_map["FOM"] = self.reductor.reconstruct(
                                q_map["ROM"], basis="parameter_basis"
                            )
                    except Exception:
                        pass

        return self.tcc_evaluator.run_multiple_models(
            models=models,
            q_map=q_map or None,
        )
    
    def IRGNM(self,
              model: InstationaryModelIP,
              q_0: VectorArray,
              alpha_0: float, 
              tol : float,
              tau : float,
              noise_level : float,
              theta: float, 
              Theta : float,
              i_max : int,
              reg_loop_max: int,
              TR_enforcement: str | None = None,
              TR_armijo_cfg: ArmijoConfig | None = None,
              lin_solver_parms: Dict = None,
              use_cached_operators: bool = False,
              dump_IRGNM_intermed_stats: bool = False,
              dump_every_nth_loop: int = 0,
              projector: SimpleBoundDomainProjector = None,
              use_error_estimator: bool = False) -> Tuple[VectorArray, Dict]: 

        assert q_0 in model.Q
        assert tol > 0
        assert tau > 0
        assert noise_level >= 0
        #assert 0 < theta < Theta < 1

        assert lin_solver_parms is not None        

        if TR_enforcement is not None:
            assert TR_enforcement in ['check_error', 'backtracking']
            assert TR_armijo_cfg is not None
            method_name = 'TR-IRGNM'
        else:
            method_name = 'IRGNM'

        self.IRGNM_statistics = {
            'IRGNM_idx' : self.IRGNM_idx,
            "q" : [],
            "alpha" : [],
            "J" : [],
            "norm_nabla_J" : [],
            "total_runtime" : [],
            "stagnation_flag" : False,
            "FOM_num_calls" : {},
            "counts" : {},
            "errors" : {
                'norm_delta_q' : [],
                'err_u' : [],
                'rel_err_u' : [],
                'err_p' : [],
                'rel_err_p' : [],
                'err_lin_u' : [],
                'rel_err_lin_u' : [],
                'err_lin_p' : [],
                'rel_err_lin_p' : [],
                'err_J' : [],
                'rel_err_J' : [],
                'err_nabla_J' : [],
                'rel_err_nabla_J' : [],
                'err_lin_J' : [],
                'rel_err_lin_J' : [],
                'err_nabla_lin_J' : [],
                'rel_err_nabla_lin_J' : []
            },
            "est_TCC" : []
        }
        
        counts = {
            'IRGNM_loop_iter' : -1,
            'reg_loop_iter' : [],
            'lin_solver_iter' : [],
            'loop_terminated' : []
        }

        stagnation_flag = False
        
        start_time = timer()
        i = 0
        tr_center_q = q_0.copy()

        model_insufficient = False
        
        alpha = alpha_0
        q = q_0.copy()
        norm_delta_q = np.sqrt(model.products['prod_Q'].apply2(q-self.last_update_q,q-self.last_update_q)[0,0])
        u, u_dot = model.solve_state(q=q, use_cached_operators=use_cached_operators, return_higher_orders=True)
        p, p_dot = model.solve_adjoint(q=q, u=u, use_cached_operators=use_cached_operators, return_higher_orders=True)
        J = model.objective(u)
        nabla_J = model.gradient(u, p, q, use_cached_operators=use_cached_operators)
        norm_nabla_J = model.compute_gradient_norm(nabla_J)
        errors = self._maybe_estimate_tr_error(
            model=model,
            previous_q=self.last_update_q,
            current_q=q,
            u=u,
            p=p,
            u_dot=u_dot,
            p_dot=p_dot,
            current_J=J,
            targets=self.error_estimate_targets_inner,
            use_cached_operators=use_cached_operators,
            use_error_estimator=use_error_estimator,
        )

        self.IRGNM_statistics["q"].append(q.copy())
        self.IRGNM_statistics["J"].append(J)
        self.IRGNM_statistics["norm_nabla_J"].append(norm_nabla_J)
        self.IRGNM_statistics["alpha"].append(alpha)
        self.IRGNM_statistics["total_runtime"].append(timer() - start_time)
        self.IRGNM_statistics["errors"]['norm_delta_q'].append(norm_delta_q)
        for key in self.IRGNM_statistics["errors"].keys():
            if key == 'norm_delta_q':
                continue
            
            if TR_enforcement is not None:
                self.IRGNM_statistics["errors"][key].append(errors.get(key, np.nan))
            else:
                self.IRGNM_statistics["errors"][key].append(np.nan)

        self.IRGNM_statistics["est_TCC"].append(self._evaluate_TCC(q))

        self.logger.debug("Running IRGNM: ")
        self.logger.debug(f"  J : {J:3.4e}")
        self.logger.debug(f"  norm_nabla_J : {norm_nabla_J:3.4e}")
        self.logger.debug(f"                ")
        self.logger.debug(f"  alpha_0 : {alpha_0:3.4e}")
        self.logger.debug(f"  tol : {tol:3.4e}")
        self.logger.debug(f"  tau : {tau:3.4e}")
        self.logger.debug(f"  theta : {theta:3.4e}")
        self.logger.debug(f"  Theta : {Theta:3.4e}")
        self.logger.debug(f"                ")
        self.logger.debug(f"  i_max : {i_max:3.4e}")
        self.logger.debug(f"  reg_loop_max : {reg_loop_max:3.4e}")


        loop_terminated = False

        x = np.zeros(shape=(model.V.dim,))
        y = np.zeros(shape=(model.V.dim,))

        while np.sqrt(2 * J) >= tol+tau*noise_level and i<i_max:
            self.logger.info(f"##############################################################################################################################")
            self.logger.warning(f"{method_name}: Iteration {i} | J = {J:3.4e} is not sufficent: {np.sqrt(2 * J):3.4e} > {(tol+tau*noise_level):3.4e}.")
            self.logger.info(f'Start {method_name} iteration {i}: J = {J:3.4e}, norm_nabla_J = {model.compute_gradient_norm(nabla_J):3.4e}, alpha = {alpha:1.4e}, |q-q^(i)| = {norm_delta_q:3.4e}')
            self.logger.info(f"------------------------------------------------------------------------------------------------------------------------------")
            self.logger.info(f"Try 1: test alpha = {alpha:3.4e}.")

            regularization_qualification = False
            projection_error_flag = False
            count = 1
            
            if projector:
                projector.pre_compute(center=q)

            d_start = q.to_numpy().copy()
            d_start[:,:] = 0
            d_start = model.Q.make_array(d_start)

            d, lin_solver_iter, projection_error_flag = self.solve_linearized_problem(
                model=model,
                q=q,
                d_start=d_start,
                alpha=alpha,
                lin_solver_parms = lin_solver_parms, 
                logger = self.logger,
                use_cached_operators=use_cached_operators,
                projector=projector
            )

            if projection_error_flag:
                self.logger.warning("Projection error while enforcing admissible domain.")
                break

            counts['lin_solver_iter'].append([lin_solver_iter])
            
            lin_u = model.solve_linearized_state(q, d, u, use_cached_operators=use_cached_operators)
            lin_J = model.linearized_objective(q, d, u, lin_u, alpha=0, use_cached_operators=use_cached_operators)

            condition_low = theta*J< 2*lin_J
            condition_up = 2* lin_J < Theta*J
            regularization_qualification = condition_low and condition_up

            if (not regularization_qualification) and (count < reg_loop_max):
                self.logger.warning(f"Used alpha = {alpha:3.4e} does NOT satisfy selection criteria: {theta*J:3.4e} < {2* lin_J:3.4e} < {Theta*J:3.4e}")
                self.logger.info(f"Searching for alpha:") 

            loop_terminated = False
            
            while (not regularization_qualification) and (count < reg_loop_max):
                count += 1

                if alpha <= 1e-14:
                    loop_terminated = True
                    break
                
                if not condition_low:
                    alpha *= 1.5  
                elif not condition_up:
                    alpha = max(alpha/2,1e-14)
                else:
                    raise ValueError
                
                self.logger.info(f"------------------------------------------------------------------------------------------------------------------------------")
                self.logger.info(f"Try {count}: test alpha = {alpha:3.4e}.")


                d, lin_solver_iter, projection_error_flag = self.solve_linearized_problem(model=model,
                                                                                          q=q,
                                                                                          d_start=d_start,
                                                                                          alpha=alpha,
                                                                                          lin_solver_parms = lin_solver_parms,
                                                                                          logger = self.logger,
                                                                                          use_cached_operators=use_cached_operators,

                                                                                          projector=projector)
                
                if projection_error_flag:
                    loop_terminated = True
                    break
                
                counts['lin_solver_iter'][-1].append(lin_solver_iter)

                lin_u = model.solve_linearized_state(q, d, u, use_cached_operators=use_cached_operators)
                lin_J = model.linearized_objective(q, d, u, lin_u, alpha=0, use_cached_operators=use_cached_operators)

                condition_low = theta*J< 2 * lin_J
                condition_up = 2* lin_J < Theta*J
                regularization_qualification = condition_low and condition_up
                            
                if (not regularization_qualification) and (count < reg_loop_max):
                    self.logger.warning(f"Used alpha = {alpha:3.4e} does NOT satisfy selection criteria: {theta*J:3.4e} < {2* lin_J:3.4e} < {Theta*J:3.4e}")
                else:
                    self.logger.info(f"------------------------------------------------------------------------------------------------------------------------------")

            loop_terminated = loop_terminated or (count >= reg_loop_max)

            counts['reg_loop_iter'].append(count)
            counts['loop_terminated'].append(loop_terminated)

            if not loop_terminated:
                self.logger.warning(f"Used alpha = {alpha:3.4e} does satisfy selection criteria: {theta*J:3.4e} < {2* lin_J:3.4e} < {Theta*J:3.4e}")
            elif loop_terminated and projection_error_flag:
                self.logger.warning("Projection error while enforcing admissible domain.")
                break
            else:   
                self.logger.error(f"Not found valid alpha before reaching maximum number of tries : {reg_loop_max}.\n\
                                   Using the last alpha tested = {alpha:3.4e}.")
                break

            ########################################### Armijo ###########################################

            TR_max_iter_cond = False
            model_insufficient = False

            if TR_enforcement == 'backtracking':
                self.logger.info(f"Enforcing TR condition using 'backtracking'.")
                q_TR, _, model_insufficient, TR_max_iter_cond, step_size, errors = self._armijo_TR_line_serach(
                    model = model,
                    previous_q = q,
                    tr_center_q = tr_center_q,
                    previous_J = J,
                    search_direction = d,
                    armijo_cfg = TR_armijo_cfg,
                    use_cached_operators=use_cached_operators,
                    projector=projector,
                    use_error_estimator=use_error_estimator
                )
                
                TR_armijo_cfg = replace(TR_armijo_cfg, initial_step_size=np.min([step_size * 2, 1]))
                self.logger.info(
                    f"Updated TR_armijo_cfg.initial_step_size = "
                    f"{TR_armijo_cfg.initial_step_size:.4e}"
                )

                if TR_max_iter_cond:
                    break

                q = q_TR

            elif TR_enforcement == 'check_error':
                self.logger.info("Enforcing TR condition using 'check_error'.")

                if projector:
                    projector.pre_compute(center=q)
                    next_q = projector.project_domain(q, d)
                else:
                    next_q = q + d

                u_r, u_dot_r = model.solve_state(q=next_q, use_cached_operators=use_cached_operators, return_higher_orders=True)
                p_r, p_dot_r = model.solve_adjoint(q=next_q, u=u, use_cached_operators=use_cached_operators, return_higher_orders=True)
                next_J = model.objective(u=u_r, q=next_q)

                # compute err only if needed by TR
                errors = self._maybe_estimate_tr_error(
                    model=model,
                    previous_q=q,
                    current_q=next_q,
                    u=u_r,
                    p=p_r,
                    u_dot=u_dot_r,
                    p_dot=p_dot_r,
                    current_J=next_J,
                    targets = self.error_estimate_targets_inner,
                    use_cached_operators=use_cached_operators,
                    use_error_estimator=use_error_estimator,
                )

                ctx = TRContext(
                    objective=float(next_J),
                    abs_error=errors.get("err_J", np.nan),
                    center_q=tr_center_q,
                    current_q=next_q,
                    product=self._tr_product(model),
                    step_size=None,
                    meta=None,
                )

                tr_res = self.TR.check(ctx)
                tr_ok = tr_res.tr_ok
                model_insufficient = tr_res.model_insufficient

                if tr_ok:
                    q = next_q

            else:
                q += d

            ########################################### Final ###########################################

            norm_delta_q = np.sqrt(model.products['prod_Q'].apply2(q-self.last_update_q,q-self.last_update_q)[0,0])
            u = model.solve_state(q, use_cached_operators=use_cached_operators)
            p = model.solve_adjoint(q, u, use_cached_operators=use_cached_operators)
            J = model.objective(u)
            nabla_J = model.gradient(u, p, q, use_cached_operators=use_cached_operators)
            norm_nabla_J = model.compute_gradient_norm(nabla_J)

            self.IRGNM_statistics["q"].append(q.copy())
            self.IRGNM_statistics["J"].append(J)
            self.IRGNM_statistics["norm_nabla_J"].append(norm_nabla_J)
            self.IRGNM_statistics["alpha"].append(alpha)

            self.IRGNM_statistics["errors"]['norm_delta_q'].append(norm_delta_q)
            for key in self.IRGNM_statistics["errors"].keys():
                if key == 'norm_delta_q':
                    continue
                
                if TR_enforcement is not None:
                    self.IRGNM_statistics["errors"][key].append(errors.get(key, np.nan))
                else:
                    self.IRGNM_statistics["errors"][key].append(np.nan)

            self.IRGNM_statistics["est_TCC"].append(self._evaluate_TCC(q))
            
            #stagnation check
            if i > 3:
                buffer = self.IRGNM_statistics["J"][-3:]
                if abs(buffer[0] - buffer[1]) / abs(buffer[0]) < STAGNATION_TOL and abs(buffer[1] - buffer[2]) / abs(buffer[1])< STAGNATION_TOL:
                    self.IRGNM_statistics["stagnation_flag"] = True
                    self.logger.info(f"Stop at iteration {i+1} of {int(i_max)}, due to stagnation.")
                    stagnation_flag = True
                    break

            self.logger.info(f'Statistics {method_name} iteration {i}: J = {J:3.4e}, norm_nabla_J = {norm_nabla_J:3.4e}, alpha = {alpha:1.4e}')
            i += 1
            if not(np.sqrt(2 * J) >= tol+tau*noise_level and i<i_max):
                self.logger.info(f"##############################################################################################################################")

            self.IRGNM_statistics["FOM_num_calls"] = self.FOM.num_calls
            if dump_IRGNM_intermed_stats:
                if self.name is not None:
                    save_path = self.save_path / f'{self.name}_IRGNM_{i}.pkl'
                else:
                    save_path = self.save_path / f'IRGNM_{i}.pkl'
                
                if (i % dump_every_nth_loop == 0) or (i == 1):
                    self.dump_stats(data=self.IRGNM_statistics,
                                    save_path=save_path)

            self.IRGNM_statistics["total_runtime"].append(timer() - start_time) 

            if model_insufficient:
                break
        
        self.logger.info(f'Final {method_name} Statistics:')
        if loop_terminated:
            self.logger.info(f'     {method_name} No sufficient regularization constant found i = {i}')
        elif i == i_max and not model_insufficient:
            self.logger.info(f'     {method_name} reached maxit at i = {i}')
        elif i < i_max and not model_insufficient:
            self.logger.info(f'     {method_name} converged at i = {i}')
        elif TR_max_iter_cond:
            self.logger.info(f'     {method_name} TR backtracking reach maximum iteration number at i = {i}')
        elif model_insufficient:
            self.logger.info(f'     {method_name} TR boundary criterium triggered at i = {i}')
        elif stagnation_flag:
            self.logger.info(f'     {method_name} TR stagnated at i = {i}')
        else:
            # Should never be happend
            raise NotImplementedError
                
        self.logger.info(f'     Start J = {self.IRGNM_statistics["J"][0]:3.4e}; Final J = {self.IRGNM_statistics["J"][-1]:3.4e}.')
        self.logger.info(f'     Start alpha = {self.IRGNM_statistics["alpha"][0]:3.4e}; Final alpha = {self.IRGNM_statistics["alpha"][-1]:3.4e}.')
        self.logger.info(f'     Start norm_nabla_J = {self.IRGNM_statistics["norm_nabla_J"][0]:3.4e}; Final norm_nabla_J = {self.IRGNM_statistics["norm_nabla_J"][-1]:3.4e}.')
        self.logger.info(f'     Euclidian distance final q and inital q = {np.linalg.norm(q.to_numpy() - q_0.to_numpy()):3.4e}')

        counts['IRGNM_loop_iter'] = self.IRGNM_idx

        self.IRGNM_statistics["counts"] = counts
        self.IRGNM_statistics["total_runtime"].append(timer() - start_time)
        self.IRGNM_idx += 1
        return (q, self.IRGNM_statistics)

    def solve_linearized_problem(self,
                                model : InstationaryModelIP, 
                                q : VectorArray,
                                d_start : VectorArray,
                                alpha : float,
                                use_cached_operators: bool,
                                logger: logging.Logger,
                                lin_solver_parms : Dict,
                                projector: SimpleBoundDomainProjector = None) -> Tuple[VectorArray, int, bool]:

        method = lin_solver_parms['method']
        if method == 'gd':
            return gradient_descent_linearized_problem(model=model,
                                                       q = q, 
                                                       d_start = d_start, 
                                                       alpha = alpha,
                                                       use_cached_operators=use_cached_operators,
                                                       logger=logger,
                                                       lin_solver_parms = lin_solver_parms,
                                                       projector=projector)
        elif method == 'BiCGSTAB':
            return BiCGStab_linearized_problem(model, 
                                               q = q, 
                                               d_start = d_start, 
                                               alpha = alpha,
                                               use_cached_operators=use_cached_operators,
                                               logger=logger,
                                               lin_solver_parms = lin_solver_parms)


           
        else:
            raise ValueError
        
    def dump_stats(self, 
                   data: Dict,
                   save_path: Union[str, Path] = None):
        if not save_path:
            save_path = self.save_path
        
        save_path = Path(save_path)
        assert save_path.suffix in ['.pkl', 'pickle']
        assert save_path.parent.exists()

        self.logger.info(f"Dumping statistics IRGNM to {save_path}.")
        save_dict_to_pkl(path=save_path, data=data, use_timestamp=False)
    
    def dump_prepare_statistics(self, statistics: Dict) -> Dict:

        for basis in statistics['reduced_bases'].keys():
            if basis in ['state_basis', 'adjoint_basis']:
                if statistics['reduced_bases'] is not None:
                    _basis = statistics['reduced_bases'][basis]
                    statistics['reduced_bases'][basis] = dealii_vector_space_to_numpy(_basis)
     
            if basis in ['state_basis', 'adjoint_basis']:
                if statistics['snapshots'] is not None:
                    snapshots = statistics['snapshots'][basis]
                    statistics['snapshots'][basis] = dealii_vector_space_to_numpy(snapshots)
            
        return statistics
          
class FOMOptimizer(Optimizer):
    def __init__(self, 
                 optimizer_parameter: Dict, 
                 FOM : InstationaryModelIP,
                 save_path : Path,
                 logger: logging.Logger = None)-> None:

        super().__init__(optimizer_parameter = optimizer_parameter, 
                         FOM = FOM, 
                         logger = logger, 
                         save_path = save_path)
        
        self.statistics = {
            "q" : [],
            'time_steps' : [],
            "alpha" : [],
            "J" : [],
            "norm_nabla_J" : [],
            "total_runtime" : [],
            "stagnation_flag" : False,
            "optimizer_parameter" : self.optimizer_parameter.copy(),
            "FOM_num_calls" : {},
            "counts" : {}
        }

    def solve(self) -> VectorArray:
        # --- schema parse/validate (TR optimizer style) ---
        cfg = FOMOptimizerCfg.from_dict(self.optimizer_parameter)

        # --- build initial q ---
        q = self.FOM.Q.make_array(cfg.q_0)
        self.last_update_q = q.copy()

        # --- initial evaluation (same as before, but uses cfg) ---
        u = self.FOM.solve_state(q, use_cached_operators=cfg.use_cached_operators)
        p = self.FOM.solve_adjoint(q, u, use_cached_operators=cfg.use_cached_operators)
        J = self.FOM.objective(u)
        nabla_J = self.FOM.gradient(u, p, q, use_cached_operators=cfg.use_cached_operators)
        norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)

        # --- centralized logging ---
        self.name = "FOM"
        log_fom_opt_config(self.logger, cfg, J=J, norm_nabla_J=norm_nabla_J)

        # --- run IRGNM (no TR enforcement here) ---
        q, IRGNM_statistic = self.IRGNM(
            model=self.FOM,
            q_0=q,
            alpha_0=cfg.alpha_0,
            tol=cfg.tol,
            tau=cfg.tau,
            noise_level=cfg.noise_level,
            theta=cfg.theta,
            Theta=cfg.Theta,
            i_max=cfg.i_max,
            reg_loop_max=cfg.reg_loop_max,
            lin_solver_parms=cfg.lin_solver_parms,
            use_cached_operators=cfg.use_cached_operators,
            dump_IRGNM_intermed_stats=True,
            dump_every_nth_loop=cfg.dump_every_nth_loop,
            projector=self.FOM_projector,
            use_error_estimator=False,  # FOM run usually doesn't need estimators
        )

        # --- store + dump statistics (unchanged semantics) ---
        self.statistics["q"] = IRGNM_statistic["q"]        
        self.statistics["alpha"] = IRGNM_statistic["alpha"]
        self.statistics["J"] = IRGNM_statistic["J"]
        self.statistics["norm_nabla_J"] = IRGNM_statistic["norm_nabla_J"]
        self.statistics["total_runtime"] = IRGNM_statistic["total_runtime"]
        self.statistics["stagnation_flag"] = IRGNM_statistic["stagnation_flag"]
        self.statistics["FOM_num_calls"] = IRGNM_statistic["FOM_num_calls"]
        self.statistics["counts"] = IRGNM_statistic["counts"]

        self.dump_stats(
            data=self.statistics,
            save_path=self.save_path / "FOM_IRGNM_final.pkl"
        )
        return q

class QrVrROMOptimizer(Optimizer):
    def __init__(self, 
                 optimizer_parameter: Dict, 
                 FOM : InstationaryModelIP,
                 save_path: Path,
                 logger: logging.Logger = None) -> None:

        super().__init__(optimizer_parameter = optimizer_parameter, 
                         FOM = FOM, 
                         logger = logger, 
                         save_path = save_path)
        
        self.QrVrROM = None

        self.use_adjoint_space = optimizer_parameter["reductor"]["use_adjoint_space"]
        if self.use_adjoint_space:
            assert optimizer_parameter['enrichment']['adjoint_basis']

        self.active_bases = []       
        for key, val in optimizer_parameter['enrichment'].items():
            assert key in ['parameter_basis','state_basis', 'adjoint_basis']            
            if val is not None:
                self.active_bases.append(key)    

        if self.use_adjoint_space:
            assert 'adjoint_basis' in self.active_bases

        reductor_cfg = InstationaryReductorConfig.from_dict(
            optimizer_parameter["reductor"],
            where="optimizer_parameter['reductor']",
            active_bases=self.active_bases,
            default_linearization_method=LinearizationMethod.DEIM,
        )

        self.reductor = build_reductor(
            FOM=FOM,
            active_bases=self.active_bases,
            config=reductor_cfg,
        )

        self.snapshot_preprocessor = SnapshotPreprocessor(
            FOM = FOM,
            active_bases = self.active_bases,
            use_adjoint_space = self.use_adjoint_space
        )

        self.all_snapshots = self.snapshot_preprocessor.make_empty_snapshots_dict()
        self.snapshots = self.snapshot_preprocessor.make_empty_snapshots_dict()
        self.initial_snapshots = self.snapshot_preprocessor.make_empty_snapshots_dict()

        self.statistics = {
            "q" : [],
            "eta" : [],
            "alpha" : [],
            "J" : [],
            "norm_nabla_J" : [],
            "J_r" : [],
            "abs_est_error_J_r" : [],
            "rel_est_error_J_r" : [],
            "abs_est_error_nabla_J_r" : [],
            "rel_est_error_nabla_J_r" : [],
            "total_runtime" : [],
            "outer_loop_runtime" : {
                'AGC_runtime' : [],
                'IRGNM_runtime' : [],
                'solve_snapshot_FOM_runtime' : [],
                'extend_runtime' : {    
                    'parameter_basis' : [],
                    'state_basis' : [],
                    'adjoint_basis' : [],
                },  
                'reduce_runtime' : [],
                'total_runtime' : []
            },
            "flags" : {
                "proj_q_in_tr" : [],
                "AGC_decay_cond" : [],
                "model_insufficient" : [],
                "check_conditions" : [],
                "rejected" : [],
            },
            "stagnation_flag" : False,
            "optimizer_parameter" : self.optimizer_parameter.copy(),
            "FOM_num_calls": {},
            "dim_Q_r" : [],
            "dim_V_r" : [],
            "counts" : [],
            "inner_loop_statistics" : [],
            "reduced_bases" : None,
            "snapshots" : None,
            "extention_stats" : {
                "snapshot_projection_error" : {
                    "parameter_basis" : [],
                    "state_basis" : []
                },
            }
        }

        if getattr(self.TR, "requires_objective_error", True):
            self.error_estimate_targets_outer.append('J')
            self.error_estimate_targets_inner.append('J')

    def extend_bases_and_rebuild_QrVrROM(
        self,
        bases: List[str],
        enrichment: Dict,
        compression_override: Optional[Dict[str, Dict]] = None,
    ) -> InstationaryModelIP:

        assert isinstance(bases, List)
        assert set(bases).issubset(self.active_bases)

        # store snapshots history
        for basis in self.snapshots.keys():
            self.all_snapshots[basis].append(self.snapshots[basis])

        for basis in bases:
            extend_start_time = timer()
            assert enrichment[basis]

            self.logger.debug(f"Extending '{basis}'")
            snapshots = self.snapshots[basis]

            base_cfg = enrichment[basis]["compression"]
            if not base_cfg:
                base_cfg = {}

            if compression_override and basis in compression_override:
                # shallow merge is enough because we're only overriding 1–2 keys
                cfg = {**base_cfg, **compression_override[basis]}
            else:
                cfg = base_cfg

            snapshots = self.snapshot_preprocessor.preprocess(
                snapshots=snapshots.copy(),
                product=self.reductor.products[basis],
                config=cfg,
            )
                
            try:
                self.reductor.extend_basis(
                    U=snapshots,
                    basis=basis,
                    method=enrichment[basis]["extend_basis"]["method"],
                    pod_modes=enrichment[basis]["extend_basis"]["pod_modes"],
                    copy_U=False,
                )
                self.reductor._check_orthonormality(basis=basis)

            except ExtensionError:
                self._logger.warning(f"No new vectors were added to '{basis}'.")

            self.statistics["outer_loop_runtime"]["extend_runtime"][basis][-1] += (
                timer() - extend_start_time
            )

        for basis in self.active_bases:
            self.reductor.dims_history[basis].append(self.reductor.get_bases_dim(basis))
        
        lengths = [len(self.reductor.dims_history[b]) for b in self.active_bases]
        assert len(set(lengths)) == 1, (
            f"Dimension histories out of sync: {dict(zip(self.active_bases, lengths))}"
        )

        self.logger.debug(f"Dim Qr-space = {self.reductor.get_bases_dim('parameter_basis')}")
        self.logger.debug(f"Dim Vr-space = {self.reductor.get_bases_dim('state_basis')}")
        self.logger.debug(f"Dim Wr-space = {self.reductor.get_bases_dim('adjoint_basis')}")

        reduce_start_time = timer()
        self.logger.debug("Creating Qr-Vr-ROM")
        QrVrROM = self.reductor.reduce()
        self.statistics["outer_loop_runtime"]["reduce_runtime"][-1] += timer() - reduce_start_time

        return QrVrROM
        
    def _reset_snapshots(self) -> None:
        self.snapshots = self.snapshot_preprocessor.make_empty_snapshots_dict()

    def _select_inner_model(
        self,
        i: int,
        schedule: Optional[List[ModelScheduleBlock]],
    ) -> InstationaryModelIP:
        if schedule is None:
            return self.QrVrROM

        period = sum(block.length for block in schedule)
        if period <= 0:
            raise ValueError("inner_loop_model_schedule period must be > 0")

        k = i % period

        acc = 0
        for block in schedule:
            acc += block.length
            if k < acc:
                if block.model == "FOM":
                    return self.FOM
                if block.model == "ROM":
                    return self.QrVrROM
                raise ValueError(f"Unknown schedule model: {block.model}")

        raise RuntimeError("Invalid inner_loop_model_schedule.")

    def add_initial_snapshots(self,
                              snapshots: VectorArray,
                              basis: str) -> None:
        
        
        assert isinstance(basis, str)
        assert basis in self.active_bases
        self.initial_snapshots[basis].append(snapshots)
        
    def solve(self) -> VectorArray:
        opt_cfg = TROptimizerCfg.from_dict(self.optimizer_parameter)

        start_time = timer()
        i = 0
        alpha = opt_cfg.alpha_0
        delta = opt_cfg.noise_level

        # ------------------------------------------------------------
        # Initial FOM evaluation
        # ------------------------------------------------------------

        solve_snapshot_FOM_start_time = timer()
        q = self.FOM.Q.make_array(opt_cfg.q_0)

        u = self.FOM.solve_state(q, use_cached_operators=False)
        p = self.FOM.solve_adjoint(q, u, use_cached_operators=False)
        J = self.FOM.objective(u)
        nabla_J, time_steps_nabla_J = self.FOM.gradient(
            u,
            p,
            q,
            use_cached_operators=opt_cfg.use_cached_operators,
            return_per_time_step=True,
        )
        norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)

        self.statistics["outer_loop_runtime"]["solve_snapshot_FOM_runtime"].append(
            timer() - solve_snapshot_FOM_start_time
        )
        for basis in self.active_bases:
            self.statistics["outer_loop_runtime"]["extend_runtime"][basis].append(0.0)
        self.statistics["outer_loop_runtime"]["reduce_runtime"].append(0.0)

        AGC_initial = min(0.5 / norm_nabla_J, 1e-3)
        AGC_armijo_cfg = replace(opt_cfg.AGC_armijo_cfg, initial_step_size=AGC_initial)

        log_tr_opt_config(self.logger, opt_cfg, J=J, norm_nabla_J=norm_nabla_J)

        # ------------------------------------------------------------
        # Initial snapshots + build initial ROM
        # ------------------------------------------------------------

        self._reset_snapshots()
        self.snapshots = self.snapshot_preprocessor.get_snapshots(
            config = opt_cfg.enrichment,
            bases = self.active_bases,
            q = q,
            u = u,
            p = p,
            nabla_J = nabla_J,
            time_steps_nabla_J = time_steps_nabla_J,
            use_cached_operators = opt_cfg.use_cached_operators
        )

        # self.FOM.A.hyperelasticity_model.save_time_series(
        #     [v.impl for v in self.snapshots['state_basis'].vectors],
        #     str('snapshots'),
        #     str(self.save_path),
        #     np.arange(len(self.snapshots['state_basis']))
        # )

        for basis in self.active_bases:
            self.snapshots[basis].append(self.initial_snapshots[basis])

        self.QrVrROM = self.extend_bases_and_rebuild_QrVrROM(
            bases=self.active_bases,
            enrichment=opt_cfg.enrichment
        )


        # self.FOM.A.hyperelasticity_model.save_time_series(
        #     [v.impl for v in self.reductor.bases['state_basis'].vectors],
        #     str('reduced_bases'),
        #     str(self.save_path),
        #     np.arange(len(self.reductor.bases['state_basis']))
        # )
        # import sys
        # sys.exit()

        # always enrich parameter basis with q and q_circ (normalized, no HaPOD)
        self._reset_snapshots()
        self.snapshots['parameter_basis'].append(q)
        self.snapshots['parameter_basis'].append(self.FOM.Q.make_array(self.FOM.setup['q_circ']))

        self.QrVrROM = self.extend_bases_and_rebuild_QrVrROM(
            bases=["parameter_basis"],
            enrichment=opt_cfg.enrichment,
            compression_override={
                "parameter_basis": {
                    "normalize": True,
                    "HaPOD": None,
                    "every_n": None,
                }
            },
        )

        self.last_update_q = self.reductor.project_vectorarray(q.copy(), 'parameter_basis')
        self.last_update_q = self.QrVrROM.Q.make_array(self.last_update_q)
        
        ############################################################

        # basis = 'state_basis'
        # _basis = self.reductor.bases[basis]

        # coeff_u = np.sum((u.inner(_basis, self.reductor.products[basis]))**2, axis=0)
        # err_i_u = np.sum(self.reductor.products[basis].pairwise_apply2(u,u)) - np.cumsum(coeff_u)
        
        # coeff_p = np.sum((p.inner(_basis, self.reductor.products[basis]))**2, axis=0)
        # err_i_p = np.sum(self.reductor.products[basis].pairwise_apply2(p,p)) - np.cumsum(coeff_p)

        # # err_i_u = err_i_u[err_i_u > 0]
        # # err_i_p = err_i_p[err_i_p > 0]

        # color = cmap(i)
        # ax_2.semilogy(err_i_u, color=color)
        # ax_2.semilogy(err_i_p, color=color, linestyle="--")
        # ax_2.set_ylim([1e-18, 1e3])
        # ax_2.grid(True)
        
        # fig_2.savefig(self.save_path / "coeffs_after_enrich.pdf")


        # ------------------------------------------------------------
        # Initial ROM evaluation at current q
        # ------------------------------------------------------------

        q_r = self.reductor.project_vectorarray(q, "parameter_basis")
        q_r = self.QrVrROM.Q.make_array(q_r)

        u_r, u_dot_r = self.QrVrROM.solve_state(
            q_r,
            use_cached_operators=opt_cfg.use_cached_operators,
            return_higher_orders=True,
        )
        
        p_r, p_dot_r = self.QrVrROM.solve_adjoint(
            q_r,
            u_r,
            use_cached_operators=opt_cfg.use_cached_operators,
            return_higher_orders=True,
        )
        J_r = self.QrVrROM.objective(u_r)
        nabla_J_r = self.QrVrROM.gradient(u_r, p_r, q_r)
        norm_nabla_J_r = self.QrVrROM.compute_gradient_norm(nabla_J_r)     

        errors = \
        self.error_evaluator.estimate_errors(
            model=self.QrVrROM,
            reductor=self.reductor,
            q_r = q_r,
            u_r = u_r,
            p_r = p_r,
            u_dot_r = u_dot_r,
            p_dot_r = p_dot_r,
            J_r = J_r,
            targets = self.error_estimate_targets_outer,
            use_cached_operators=opt_cfg.use_cached_operators,
            use_error_estimator=opt_cfg.use_error_estimator
        )

    
        abs_est_error_J_r = OBJ.sanitize_error(errors.get("err_J", np.nan), name="err_J")
        rel_est_error_J_r = OBJ.rel_error(abs_error = abs_est_error_J_r, objective=J_r)

        abs_est_error_nabla_J_r = OBJ.sanitize_error(errors.get("err_nabla_J", np.nan), name="err_nabla_J")
        rel_est_error_nabla_J_r = OBJ.rel_error(abs_error = abs_est_error_nabla_J_r, objective=norm_nabla_J_r)

        self.statistics["q"].append(q)
        self.statistics["eta"].append(self.TR.eta)
        self.statistics["alpha"].append(alpha)
        self.statistics["J"].append(J)
        self.statistics["norm_nabla_J"].append(norm_nabla_J)
        self.statistics["J_r"].append(J_r)
        self.statistics['abs_est_error_J_r'].append(abs_est_error_J_r)
        self.statistics['rel_est_error_J_r'].append(rel_est_error_J_r)
        self.statistics['abs_est_error_nabla_J_r'].append(abs_est_error_nabla_J_r)
        self.statistics['rel_est_error_nabla_J_r'].append(rel_est_error_nabla_J_r)
        self.statistics['dim_Q_r'].append(self.reductor.get_bases_dim('parameter_basis'))
        self.statistics['dim_V_r'].append(self.reductor.get_bases_dim('state_basis'))

        # ------------------------------------------------------------
        # Outer loop
        # ------------------------------------------------------------
        convergence_criterium = np.sqrt(2 * J) < opt_cfg.tol + opt_cfg.tau * opt_cfg.noise_level
        AGC_jump_back = False
        last_inner_alpha = None
        IRGNM_statistics = {}

        while (not convergence_criterium) and (i < opt_cfg.i_max):
            outer_loop_start_time = timer()
            self.logger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            self.logger.warning(
                f"Qr-Vr-IRGNM iteration {i}: J = {J:3.4e} is not sufficent: {np.sqrt(2 * J):3.4e} > "
                f"{(opt_cfg.tol + opt_cfg.tau * opt_cfg.noise_level):3.4e}."
            )
            self.logger.info(
                f"Start Qr-Vr-IRGNM iteration {i}: J = {J:3.4e}, norm_nabla_J = {norm_nabla_J:3.4e}, alpha = {alpha:1.4e}"
            )
            self.logger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            
            assert self.FOM_projector.project_domain(center=q) == q
            
            if self.TR.eta_too_small():
                self.statistics["stagnation_flag"] = True
                break
            
            # current reduced coordinate
            q_r = self.reductor.project_vectorarray(q, "parameter_basis")
            q_r = self.QrVrROM.Q.make_array(q_r)

            u_r, u_dot_r = self.QrVrROM.solve_state(
                q_r,
                use_cached_operators=opt_cfg.use_cached_operators,
                return_higher_orders=True,
            )
            p_r, p_dot_r = self.QrVrROM.solve_adjoint(
                q_r,
                u_r,
                use_cached_operators=opt_cfg.use_cached_operators,
                return_higher_orders=True,
            )
            
            J_r = self.QrVrROM.objective(u_r)
            nabla_J_r = self.QrVrROM.gradient(
                u_r, p_r, q_r, use_cached_operators=opt_cfg.use_cached_operators
            )
            norm_nabla_J_r = self.QrVrROM.compute_gradient_norm(nabla_J_r)

            ########################################### q^{(i)} in TR ###########################################

            errors = self._maybe_estimate_tr_error(
                model=self.QrVrROM,
                previous_q=q_r,
                current_q=q_r,
                u=u_r,
                p=p_r,
                u_dot=u_dot_r,
                p_dot=p_dot_r,
                current_J=J_r,
                targets=self.error_estimate_targets_outer,
                use_cached_operators=opt_cfg.use_cached_operators,
                use_error_estimator=opt_cfg.use_error_estimator,
            )
            
            ctx = TRContext(
                objective=float(J_r),
                abs_error=float(errors.get("err_J", np.nan)),
                center_q=self.last_update_q,
                current_q=q_r,
                product=self._tr_product(self.QrVrROM),
                step_size=None,
                meta=None,
            )

            tr_center = self.TR.check(ctx)
            proj_q_in_tr = tr_center.tr_ok

            if AGC_jump_back: 
                self.statistics['flags']['proj_q_in_tr'][-1] = proj_q_in_tr
            else:
                self.statistics['flags']['proj_q_in_tr'].append(proj_q_in_tr)

            if not proj_q_in_tr:
                self._logger.warning(f"q^(i) is not in the trust region.")
                self._logger.warning(f"Extending reduced spaces with all snapshots.")

                self._reset_snapshots()
                self.snapshots = self.snapshot_preprocessor.get_snapshots(
                    config = opt_cfg.enrichment,
                    bases = self.active_bases,
                    q = q,
                    u = u,
                    p = p,
                    nabla_J = nabla_J,
                    time_steps_nabla_J = None,
                    add_additional_snapshots = False,
                    use_cached_operators = opt_cfg.use_cached_operators
                )

                self.QrVrROM = self.extend_bases_and_rebuild_QrVrROM(
                    bases=self.active_bases,
                    enrichment=opt_cfg.enrichment,
                    compression_override={
                        basis: {
                            "normalize": True,
                            "HaPOD": None,
                            "every_n": None,
                        }
                        for basis in self.active_bases
                    },
                )   

                self.last_update_q = self.reductor.project_vectorarray(q.copy(), 'parameter_basis')
                self.last_update_q = self.QrVrROM.Q.make_array(self.last_update_q)
                
                self.FOM.reset_cached_operators()

                q_r = self.reductor.project_vectorarray(q, "parameter_basis")
                q_r = self.QrVrROM.Q.make_array(q_r)

                u_r, u_dot_r = self.QrVrROM.solve_state(
                    q_r, use_cached_operators=False, return_higher_orders=True
                )
                p_r, p_dot_r = self.QrVrROM.solve_adjoint(
                    q_r, u_r, use_cached_operators=False, return_higher_orders=True
                )

                J_r = self.QrVrROM.objective(u_r)
                nabla_J_r = self.QrVrROM.gradient(u_r, p_r, q_r)
                norm_nabla_J_r = self.QrVrROM.compute_gradient_norm(nabla_J_r)

                errors = self._maybe_estimate_tr_error(
                    model=self.QrVrROM,
                    previous_q=q_r,
                    current_q=q_r,
                    u=u_r,
                    p=p_r,
                    u_dot=u_dot_r,
                    p_dot=p_dot_r,
                    current_J=J_r,
                    targets=self.error_estimate_targets_outer,
                    use_cached_operators=opt_cfg.use_cached_operators,
                    use_error_estimator=opt_cfg.use_error_estimator,
                )
                
                ctx = TRContext(
                    objective=float(J_r),
                    abs_error=float(errors.get("err_J", np.nan)),
                    center_q=self.last_update_q,
                    current_q=q_r,
                    product=self._tr_product(self.QrVrROM),
                    step_size=None,
                    meta=None,
                )

                tr_center = self.TR.check(ctx)

            tr_center = self.TR.check(ctx)
            proj_q_in_tr = tr_center.tr_ok
            assert proj_q_in_tr 

            projector = SimpleBoundDomainProjector(
                model = self.QrVrROM,
                bounds = self.FOM.bounds,
                reductor = self.reductor,
                use_sufficient_condition = True,
                #use_sufficient_condition = False,
                logger = self.logger
            )
            #projector = None

            # ------------------------------------------------------------
            # AGC with Armijo+TR backtracking
            # ------------------------------------------------------------ 
            self.logger.warning("Calculate AGC with Armijo backtracking.")

            AGC_start_time = timer()

            if opt_cfg.reg_AGC_step:                
                if last_inner_alpha is not None:
                    AGC_alpha = last_inner_alpha
                else:
                    AGC_alpha = alpha

                previous_J = self.QrVrROM.objective(u_r, q=q_r, alpha=AGC_alpha)
                nabla_reg_J_r = self.QrVrROM.gradient(u_r, p_r, q_r, alpha=AGC_alpha)
                norm_grad = self.QrVrROM.compute_gradient_norm(nabla_reg_J_r)            
                search_direction = -nabla_reg_J_r
                search_direction.scal(1.0 / norm_grad)
            else:
                AGC_alpha = 0.0
                previous_J = J_r
                norm_grad = self.QrVrROM.compute_gradient_norm(nabla_J_r)            
                search_direction = -nabla_J_r
                search_direction.scal(1.0 / norm_grad)

            self.logger.warning(f"Using AGC_alpha = {AGC_alpha}.")

            q_AGC, J_r_AGC, model_insufficient, max_iter_cond_AGC, _, errors_AGC = self._armijo_TR_line_serach(
                model = self.QrVrROM,
                previous_q = q_r,
                tr_center_q=self.last_update_q,
                previous_J = previous_J,
                search_direction = search_direction,
                armijo_cfg = AGC_armijo_cfg,
                use_cached_operators=opt_cfg.use_cached_operators,
                projector = projector,
                alpha = AGC_alpha,
                use_error_estimator=opt_cfg.use_error_estimator
            )
            
            AGC_decay_cond = J_r_AGC < (J + MACHINE_EPS)
            
            if not AGC_jump_back:
                self.statistics['flags']['AGC_decay_cond'].append(AGC_decay_cond)

            if opt_cfg.reg_AGC_step:
                AGC_decay_cond = True
                        
            if not AGC_decay_cond:
                self._logger.warning(f"J_r_AGC = {J_r_AGC:3.4e} is greater or equal than J = {J:3.4e}.")
                self._logger.warning(f"Extending reduced spaces with all snapshots and recomputing AGC.")

                # This is correct, since using u and p with cached values lead to an error.
                # TODO Figure out why caching here prodcues wrong results.
                u = self.FOM.solve_state(q, use_cached_operators=False)        
                p = self.FOM.solve_adjoint(q, u, use_cached_operators=False)

                self._reset_snapshots()
                self.snapshots = self.snapshot_preprocessor.get_snapshots(
                    config = opt_cfg.enrichment,
                    bases = self.active_bases,
                    q = q,
                    u = u,
                    p = p,
                    nabla_J = nabla_J,
                    time_steps_nabla_J = None,
                    add_additional_snapshots = False,
                    use_cached_operators = opt_cfg.use_cached_operators
                )

                self.QrVrROM = self.extend_bases_and_rebuild_QrVrROM(
                    bases=self.active_bases,
                    enrichment=opt_cfg.enrichment,
                    compression_override={
                        basis: {
                            "normalize": True,
                            "HaPOD": None,
                            "every_n" : None,
                        }
                        for basis in self.active_bases
                    },
                )   

                self.last_update_q = self.reductor.project_vectorarray(q.copy(), 'parameter_basis')
                self.last_update_q = self.QrVrROM.Q.make_array(self.last_update_q)

                AGC_jump_back = True
                self.TR.shrink()
                continue
            
            if not opt_cfg.reg_AGC_step:
                assert not max_iter_cond_AGC

            AGC_jump_back = False
            self.statistics['flags']['model_insufficient'].append(model_insufficient)
            self.statistics["outer_loop_runtime"]['AGC_runtime'].append(timer() - AGC_start_time)

            self.statistics['flags']['model_insufficient'].append(model_insufficient)
            self.statistics["outer_loop_runtime"]['AGC_runtime'].append(timer() - AGC_start_time)

            q_r = q_AGC.copy()

            # ------------------------------------------------------------
            # IRGNM inner loop
            # ------------------------------------------------------------
            IRGNM_start_time = timer()

            model = self._select_inner_model(
                i=i,
                schedule=opt_cfg.inner_loop_model_schedule
            )

            # ----------------------------
            # Identify model + log
            # ----------------------------
            if model is self.FOM:
                model_name = "FOM"
            elif model is self.QrVrROM:
                model_name = "ROM"
            else:
                raise ValueError("Unknown model returned by _select_inner_model")


            # ----------------------------
            # Prepare inputs
            # ----------------------------

            do_inner_loop = not model_insufficient
            q_ = q_r.copy()
            projector_ = projector
            i_max_inner_ = opt_cfg.i_max_inner

            if model is self.FOM:
                do_inner_loop = True
                q_ = self.reductor.reconstruct(
                    x=q_r.copy(),
                    basis='parameter_basis'
                )
                projector_ = self.FOM_projector
                
                self.last_update_q = self.reductor.reconstruct(
                    x=self.last_update_q.copy(),
                    basis='parameter_basis'
                )
                i_max_inner_ = 1                

            # ----------------------------
            # Run IRGNM
            # ----------------------------
            if do_inner_loop:
                self.logger.info(f"[Outer {i}] Model={model_name}, do_inner_loop={do_inner_loop}")

                q__, IRGNM_statistic = self.IRGNM(
                    model=model,
                    q_0=q_,
                    alpha_0=alpha,
                    tol=opt_cfg.tol,
                    tau=opt_cfg.tau,
                    noise_level=delta,
                    i_max=i_max_inner_,
                    theta=opt_cfg.theta,
                    Theta=opt_cfg.Theta,
                    reg_loop_max=opt_cfg.reg_loop_max,
                    TR_enforcement=opt_cfg.TR_enforcement,
                    TR_armijo_cfg=opt_cfg.TR_armijo_cfg,
                    lin_solver_parms=opt_cfg.lin_solver_parms,
                    use_cached_operators=opt_cfg.use_cached_operators,
                    projector=projector_,
                    use_error_estimator=opt_cfg.use_error_estimator,
                )

            if model is self.FOM:                
                q_r = self.QrVrROM.Q.make_array(
                    self.reductor.project_vectorarray(q__.copy(), "parameter_basis")
                )
            elif model is self.QrVrROM:
                q_r = q__.copy()

            
            self.statistics["outer_loop_runtime"]['IRGNM_runtime'].append(timer() - IRGNM_start_time)

            # ------------------------------------------------------------
            # Accept / Reject
            # ------------------------------------------------------------

            check_conditions = bool(IRGNM_statistic) and (len(IRGNM_statistic.get("q", [])) > 1)
            if model is self.FOM:
                check_conditions = False

            self.statistics['flags']['check_conditions'].append(check_conditions)

            if check_conditions:
                self.logger.debug("Decide on q; Either accept or reject")

                u_r, u_dot_r = self.QrVrROM.solve_state(
                    q_r, use_cached_operators=opt_cfg.use_cached_operators, return_higher_orders=True
                )
                p_r, p_dot_r = self.QrVrROM.solve_adjoint(
                    q_r, u_r, use_cached_operators=opt_cfg.use_cached_operators, return_higher_orders=True
                )

                J_r = self.QrVrROM.objective(u_r)
                nabla_J_r = self.QrVrROM.gradient(u_r, p_r, q_r)
                norm_nabla_J_r = self.QrVrROM.compute_gradient_norm(nabla_J_r)

                errors = self.error_evaluator.estimate_errors(
                    model=self.QrVrROM,
                    reductor=self.reductor,
                    q_r=q_r,
                    u_r=u_r,
                    p_r=p_r,
                    u_dot_r=u_dot_r,
                    p_dot_r=p_dot_r,
                    J_r=J_r,
                    targets=self.error_estimate_targets_outer,
                    use_cached_operators=opt_cfg.use_cached_operators,
                    use_error_estimator=opt_cfg.use_error_estimator,
                )

                abs_est_error_J_r = OBJ.sanitize_error(errors.get("err_J", np.nan), name="err_J")
                rel_est_error_J_r = OBJ.rel_error(abs_error = abs_est_error_J_r, objective=J_r)

                abs_est_error_nabla_J_r = OBJ.sanitize_error(errors.get("err_nabla_J", np.nan), name="err_nabla_J")
                rel_est_error_nabla_J_r = OBJ.rel_error(abs_error = abs_est_error_nabla_J_r, objective=norm_nabla_J_r)

                sufficent_condition = (J_r + abs_est_error_J_r) < J_r_AGC
                necessary_condition = (J_r - abs_est_error_J_r) <= J_r_AGC

                self.logger.debug(f"    J_r_AGC = {J_r_AGC:3.4e}")
                self.logger.debug(f"    J_r = {J_r:3.4e}")
                self.logger.debug(f"    abs_est_error_J_r = {abs_est_error_J_r:3.4e}")
                self.logger.debug(
                    f"    J_r + abs_est_error_J_r = {(J_r + abs_est_error_J_r):3.4e}; sufficent_condition = {sufficent_condition}"
                )
                self.logger.debug(
                    f"    J_r - abs_est_error_J_r = {(J_r - abs_est_error_J_r):3.4e}; necessary_condition = {necessary_condition}"
                )

                rejected = False
                
                if sufficent_condition:
                    self.logger.info("    Accept q.")
                    rejected = False

                    solve_snapshot_FOM_start_time = timer()
                    q = self.reductor.reconstruct(q_r, basis="parameter_basis")
                    q = self.FOM_projector.project_domain(center=q)

                    u = self.FOM.solve_state(q, use_cached_operators=opt_cfg.use_cached_operators)
                    p = self.FOM.solve_adjoint(q, u, use_cached_operators=opt_cfg.use_cached_operators)
                    J = self.FOM.objective(u)
                    nabla_J, time_steps_nabla_J = self.FOM.gradient(
                        u,
                        p,
                        q,
                        use_cached_operators=opt_cfg.use_cached_operators,
                        return_per_time_step=True,
                    )
                    norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)
                    self.statistics["outer_loop_runtime"]["solve_snapshot_FOM_runtime"].append(
                        timer() - solve_snapshot_FOM_start_time
                    )

                    self.TR.update_by_trustworthiness(
                        obj_r=J_r,
                        obj_r_center=self.statistics["J_r"][-1],
                        obj=J,
                        obj_center=self.statistics["J"][-1],
                    )
                elif not necessary_condition:
                    self.logger.info("    Reject q.")
                    rejected = True
                    self.TR.shrink()

                    solve_snapshot_FOM_start_time = timer()
                    self.statistics["outer_loop_runtime"]["solve_snapshot_FOM_runtime"].append(
                        timer() - solve_snapshot_FOM_start_time
                    )

                else:
                    solve_snapshot_FOM_start_time = timer()
                    q_ = self.reductor.reconstruct(q_r, basis="parameter_basis")
                    q_ = self.FOM_projector.project_domain(center=q_)

                    u_ = self.FOM.solve_state(q_, use_cached_operators=opt_cfg.use_cached_operators)
                    p_ = self.FOM.solve_adjoint(q_, u_, use_cached_operators=opt_cfg.use_cached_operators)
                    J_ = self.FOM.objective(u_)

                    nabla_J_, time_steps_nabla_J_ = self.FOM.gradient(
                        u_,
                        p_,
                        q_,
                        use_cached_operators=opt_cfg.use_cached_operators,
                        return_per_time_step=True,
                    )
                    norm_nabla_J_ = self.FOM.compute_gradient_norm(nabla_J_)
                    self.statistics["outer_loop_runtime"]["solve_snapshot_FOM_runtime"].append(
                        timer() - solve_snapshot_FOM_start_time
                    )

                    EASDC = J_ <= J_r_AGC
                    self.logger.info(f"    J = {J:3.4e}; EASDC = {EASDC}.")

                    if EASDC:
                        self.logger.info("    Accept q.")
                        rejected = False

                        q, u, p, J = q_, u_, p_, J_
                        nabla_J, time_steps_nabla_J = nabla_J_, time_steps_nabla_J_
                        norm_nabla_J = norm_nabla_J_

                        self.TR.update_by_trustworthiness(
                            obj_r=J_r,
                            obj_r_center=self.statistics["J"][-1],
                            obj=J,
                            obj_center=self.statistics["J_r"][-1],
                        )
                    else:
                        self.logger.info("    Reject q.")
                        rejected = True
                        self.TR.shrink()

                    self.logger.info(f"    eta = {self.TR.eta:3.4e}.") 
            else:
                if model is self.FOM:
                    self.logger.debug("FOM used; Skipping check.")
                else:    
                    self.logger.debug("Not found q_trial; Using AGC.")
                    q_r = q_AGC.copy()
                
                rejected = False

                solve_snapshot_FOM_start_time = timer()
                q = self.reductor.reconstruct(q_r, basis="parameter_basis")
                q = self.FOM_projector.project_domain(center=q)

                u = self.FOM.solve_state(q, use_cached_operators=opt_cfg.use_cached_operators)
                p = self.FOM.solve_adjoint(q, u, use_cached_operators=opt_cfg.use_cached_operators)
                J = self.FOM.objective(u)

                self.statistics["outer_loop_runtime"]["solve_snapshot_FOM_runtime"].append(
                    timer() - solve_snapshot_FOM_start_time
                )

                nabla_J, time_steps_nabla_J = self.FOM.gradient(
                    u,
                    p,
                    q,
                    use_cached_operators=opt_cfg.use_cached_operators,
                    return_per_time_step=True,
                )
                norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)
                self.TR.shrink()

                #For the statistics
                nabla_J_r = self.QrVrROM.compute_gradient(
                    q=q_r,
                    alpha=alpha,
                    use_cached_operators = opt_cfg.use_cached_operators
                )
                norm_nabla_J_r = self.QrVrROM.compute_gradient_norm(nabla_J_r)

                abs_est_error_J_r = OBJ.sanitize_error(errors_AGC.get("err_J", np.nan), name="err_J")
                rel_est_error_J_r = OBJ.rel_error(abs_error = abs_est_error_J_r, objective=J_r_AGC)

                abs_est_error_nabla_J_r = OBJ.sanitize_error(errors_AGC.get("err_nabla_J", np.nan), name="err_nabla_J")
                rel_est_error_nabla_J_r = OBJ.rel_error(abs_error = abs_est_error_nabla_J_r, objective=norm_nabla_J_r)


            for basis in self.active_bases:
                self.statistics["outer_loop_runtime"]["extend_runtime"][basis].append(0.0)
            self.statistics["outer_loop_runtime"]["reduce_runtime"].append(0.0)

            # ------------------------------------------------------------
            # Final bookkeeping / enrichment if accepted
            # ------------------------------------------------------------

            convergence_criterium = np.sqrt(2 * J) < opt_cfg.tol + opt_cfg.tau * opt_cfg.noise_level
            self.statistics["flags"]["rejected"].append(rejected)

            if not rejected:
                # keep delta as-is (original behavior)
                delta = delta

                if IRGNM_statistic:
                    try:
                        alpha = IRGNM_statistic["alpha"][1]
                        last_inner_alpha = IRGNM_statistic["alpha"][-1]
                    except Exception:
                        last_inner_alpha = None

                if not convergence_criterium:

                    # print("Called")
                    # self.reductor.delete_cached_operators()

                    # self.reductor.bases["state_basis"] = self.FOM.V.empty()
                    
                    self._reset_snapshots()
                    self.snapshots = self.snapshot_preprocessor.get_snapshots(
                        config=opt_cfg.enrichment,
                        bases=self.active_bases,
                        q=q,
                        u=u,
                        p=p,
                        nabla_J=nabla_J,
                        time_steps_nabla_J=time_steps_nabla_J,
                        use_cached_operators=opt_cfg.use_cached_operators,
                    )

                    # (keep your existing coarsing blocks here as-is; omitted in your snippet refactor scope)

                    # self.reductor.delete_cached_operators()
                    # self.reductor._cached_operators = {
                    #     "A": None,
                    #     "A_r_state": None,
                    #     "A_r_adjoint": None,
                    #     "A_r_adjoint_state": None,
                    # }
                    # self.reductor.bases["parameter_basis"] = self.FOM.Q.empty()
                    # self.reductor.bases["state_basis"] = self.FOM.V.empty()


                    # u_r = self.QrVrROM.solve_state(
                    #     q_r
                    # )

                    # diff = u - self.reductor.reconstruct(u_r, basis="state_basis")                    
                    # self.FOM.A.hyperelasticity_model.save_time_series(
                    #     [v.impl for v in diff.vectors],
                    #     str('diff_u'),
                    #     str(self.save_path / f'{i}'),
                    #     np.linspace(self.FOM.T_initial, self.FOM.T_final, self.FOM.nt+1)
                    # )

                    # self.FOM.A.hyperelasticity_model.save_time_series(
                    #     [v.impl for v in u.vectors],
                    #     str('u'),
                    #     str(self.save_path / f'{i}'),
                    #     np.linspace(self.FOM.T_initial, self.FOM.T_final, self.FOM.nt+1)
                    # )
                        

                    # import sys
                    # sys.exit()
                                        
                    self.QrVrROM = self.extend_bases_and_rebuild_QrVrROM(
                        bases=self.active_bases,
                        enrichment=opt_cfg.enrichment,
                    )

                    # self._reset_snapshots()
                    # self.snapshots['parameter_basis'].append(q)
                    # self.snapshots['parameter_basis'].append(self.FOM.Q.make_array(self.FOM.setup['q_circ']))

                    # self.QrVrROM = self.extend_bases_and_rebuild_QrVrROM(
                    #     bases=["parameter_basis"],
                    #     enrichment=opt_cfg.enrichment,
                    #     compression_override={
                    #         "parameter_basis": {
                    #             "normalize": True,
                    #             "HaPOD": None,
                    #             "every_n": None,
                    #         }
                    #     },
                    # )


                    self.last_update_q = self.reductor.project_vectorarray(q.copy(), "parameter_basis")
                    self.last_update_q = self.QrVrROM.Q.make_array(self.last_update_q)

                    q_r = self.reductor.project_vectorarray(q, "parameter_basis")
                    q_r = self.QrVrROM.Q.make_array(q_r)

                self.statistics["q"].append(q)
                self.statistics["eta"].append(self.TR.eta)
                self.statistics["alpha"].append(alpha)
                self.statistics["J"].append(J)
                self.statistics["norm_nabla_J"].append(norm_nabla_J)
                self.statistics["J_r"].append(J_r)
                self.statistics["abs_est_error_J_r"].append(abs_est_error_J_r)
                self.statistics["rel_est_error_J_r"].append(rel_est_error_J_r)
                self.statistics["abs_est_error_nabla_J_r"].append(abs_est_error_nabla_J_r)
                self.statistics["rel_est_error_nabla_J_r"].append(rel_est_error_nabla_J_r)
                self.statistics["dim_Q_r"].append(self.reductor.get_bases_dim("parameter_basis"))
                self.statistics["dim_V_r"].append(self.reductor.get_bases_dim("state_basis"))

                self.statistics["counts"].append(IRGNM_statistic.get("counts", {}))
                self.statistics["inner_loop_statistics"].append(IRGNM_statistic)
                self.statistics["total_runtime"].append(timer() - start_time)
                self.statistics["outer_loop_runtime"]["total_runtime"].append(
                    timer() - outer_loop_start_time
                )
                self.statistics["FOM_num_calls"] = self.FOM.num_calls

            if (i % opt_cfg.dump_every_nth_loop == 0) or (i == 1):
                self.dump_stats(
                    data=self.statistics,
                    save_path=self.save_path / f"TR_IRGNM_{i}.pkl",
                )

            if i > 3:
                buffer = self.statistics["J"][-3:]
                if (
                    abs(buffer[0] - buffer[1]) / abs(buffer[0]) < STAGNATION_TOL
                    and abs(buffer[1] - buffer[2]) / abs(buffer[1]) < STAGNATION_TOL
                ):
                    self.statistics["stagnation_flag"] = True
                    self.logger.info(
                        f"Stop at iteration {i+1} of {int(opt_cfg.i_max)}, due to stagnation."
                    )
                    break

            if convergence_criterium:
                break

            i += 1

        self.statistics["FOM_num_calls"] = self.FOM.num_calls
        self.statistics["reduced_bases"] = self.reductor.bases

        data = self.dump_prepare_statistics(self.statistics)
        self.dump_stats(data=data, save_path=self.save_path / "TR_IRGNM_final.pkl")

        return q

