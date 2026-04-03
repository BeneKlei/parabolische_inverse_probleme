# RBInvParam/optimizer/error_evaluator.py
# ============================================================
# ErrorEvaluator: computes error metrics either via
#   (a) ROM error *estimation* (a posteriori estimator), or
#   (b) "truth" error computation using FOM reconstructions.
#
# Public API:
#   errors = evaluator.compute_errors(...)
#
# Returns a dict with a consistent set of keys; values are np.nan
# if not computed / not requested / not implemented.
# ============================================================

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Union, Any

from pymor.vectorarrays.interface import VectorArray

from RBInvParam.model import InstationaryModelIP
from RBInvParam.reduction.base import BaseIPReductor


class ErrorEvaluator:
    """
    Owns all error computation:
      - compute_errors (public)
      - _estimate_error (estimator-based, e.g. ROM residual estimator)
      - _calc_errors (truth-based, reconstruct and compare to FOM)

    Intended to be used by optimizers:
        self.errors = ErrorEvaluator(FOM=self.FOM, logger=self.logger)
        errors = self.errors.compute_errors(...)
    """

    def __init__(self, *, FOM: InstationaryModelIP, logger=None):
        self.FOM = FOM
        self.logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def estimate_errors(
        self,
        *,
        model: InstationaryModelIP,
        reductor: Optional[BaseIPReductor],
        q_r: VectorArray,
        d_r: Optional[VectorArray] = None,
        u_r: Optional[VectorArray] = None,
        p_r: Optional[VectorArray] = None,
        u_dot_r: Optional[VectorArray] = None,
        p_dot_r: Optional[VectorArray] = None,
        lin_u_r: Optional[VectorArray] = None,
        lin_p_r: Optional[VectorArray] = None,
        J_r: Optional[float] = None,
        targets: Union[str, List[str]] = "all",
        use_error_estimator: bool = True,
        use_cached_operators: bool = True,
    ) -> Dict[str, Any]:
        """
        Always returns a dict with consistent keys.
        - If use_error_estimator=True: uses model's estimator (e.g. ROM a posteriori)
        - Else: computes "true" errors via reconstructing to FOM and comparing

        NOTE: For FOM itself, "error vs FOM" is undefined; returns NaNs.
        """

        if id(model) == id(self.FOM):
            return self._empty_error_dict()

        if use_error_estimator:
            return self._estimate_error(
                model=model,
                q_r=q_r,
                d_r=d_r,
                u_r=u_r,
                p_r=p_r,
                u_dot_r=u_dot_r,
                p_dot_r=p_dot_r,
                lin_u_r=lin_u_r,
                lin_p_r=lin_p_r,
                J_r=J_r,
                targets=targets,
                use_cached_operators=use_cached_operators,
            )

        if reductor is None:
            raise ValueError("reductor must be provided when use_error_estimator=False")

        return self._calc_errors(
            model=model,
            reductor=reductor,
            q_r=q_r,
            d_r=d_r,
            u_r=u_r,
            p_r=p_r,
            lin_u_r=lin_u_r,
            lin_p_r=lin_p_r,
            targets=targets,
            use_cached_operators=use_cached_operators,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_error_dict() -> Dict[str, Any]:
        return {
            "err_u": np.nan, "rel_err_u": np.nan,
            "err_p": np.nan, "rel_err_p": np.nan,
            "err_lin_u": np.nan, "rel_err_lin_u": np.nan,
            "err_lin_p": np.nan, "rel_err_lin_p": np.nan,
            "err_J": np.nan, "rel_err_J": np.nan,
            "err_nabla_J": np.nan, "rel_err_nabla_J": np.nan,
            "err_lin_J": np.nan, "rel_err_lin_J": np.nan,
            "err_nabla_lin_J": np.nan, "rel_err_nabla_lin_J": np.nan,
        }

    # ------------------------------------------------------------------
    # Estimator-based errors (was Optimizer._estimate_error)
    # ------------------------------------------------------------------
    def _estimate_error(
        self,
        *,
        model: InstationaryModelIP,
        q_r: VectorArray,
        d_r: Optional[VectorArray] = None,
        u_r: Optional[VectorArray] = None,
        p_r: Optional[VectorArray] = None,
        u_dot_r: Optional[VectorArray] = None,
        p_dot_r: Optional[VectorArray] = None,
        lin_u_r: Optional[VectorArray] = None,
        lin_p_r: Optional[VectorArray] = None,
        J_r: Optional[float] = None,
        targets: Union[str, List[str]] = "all",
        use_cached_operators: bool = True,
    ) -> Dict[str, Any]:

        ordered_targets = ["u", "p", "lin_u", "lin_p", "J", "nabla_J", "lin_J", "nabla_lin_J"]
        implemented_targets = ["J"]  # keep your current estimator behavior

        if targets == "all":
            targets = ordered_targets

        assert set(targets).issubset(implemented_targets)

        out = self._empty_error_dict()

        for target in targets:
            if target == "J":
                assert q_r is not None
                assert u_r is not None
                assert u_dot_r is not None
                assert J_r is not None
                assert J_r > 0

                est_err_J = model.estimate_objective_error(
                    q=q_r,
                    u=u_r,
                    u_dot=u_dot_r,
                    J=J_r,
                    use_cached_operators=use_cached_operators,
                )

                if self.logger:
                    self.logger.debug(f"Estimated err_J = {est_err_J:3.4e}")
                    self.logger.debug(f"Estimated rel_err_J = {(est_err_J / J_r):3.4e}")

                out["err_J"] = float(est_err_J)
                out["rel_err_J"] = float(est_err_J / J_r)
                continue

            raise ValueError(f"Target not implemented in estimator: {target}")

        return out

    # ------------------------------------------------------------------
    # Truth-based errors (was Optimizer._calc_errors)
    # ------------------------------------------------------------------
    def _calc_errors(
        self,
        *,
        model: InstationaryModelIP,
        reductor: BaseIPReductor,
        q_r: VectorArray,
        d_r: Optional[VectorArray] = None,
        u_r: Optional[VectorArray] = None,
        p_r: Optional[VectorArray] = None,
        lin_u_r: Optional[VectorArray] = None,
        lin_p_r: Optional[VectorArray] = None,
        targets: Union[str, List[str]] = "all",
        use_cached_operators: bool = True,
    ) -> Dict[str, Any]:

        ordered_targets = ["u", "p", "lin_u", "lin_p", "J", "nabla_J", "lin_J", "nabla_lin_J"]
        out = self._empty_error_dict()

        if targets == "all":
            targets = ordered_targets

        if len(targets) == 0:
            return out

        assert set(targets).issubset(set(ordered_targets))

        if set(targets).issubset({"lin_u", "lin_p", "lin_J", "nabla_lin_J"}):
            assert d_r is not None


        # reconstruct reduced q (and d) to full space
        q = reductor.reconstruct(q_r, basis="parameter_basis")
        if d_r is not None:
            d = reductor.reconstruct(d_r, basis="parameter_basis")

        u = p = lin_u = lin_p = None

        J_r_ = None
        nabla_J_r_ = None
        lin_J_r_ = None
        nabla_lin_J_r_ = None

        required = []
        for t in targets:
            if t == "u":
                required += ["u"]
            elif t == "p":
                required += ["u", "p"]
            elif t == "lin_u":
                required += ["u", "lin_u"]
            elif t == "lin_p":
                required += ["lin_u", "lin_p"]
            elif t == "J":
                required += ["u", "J"]
            elif t == "nabla_J":
                required += ["u", "p", "nabla_J"]
            elif t == "lin_J":
                required += ["lin_u", "lin_J"]
            elif t == "nabla_lin_J":
                required += ["u", "lin_p", "nabla_lin_J"]

        required = [x for x in ordered_targets if x in required]

        for rq in required:
            if rq == "u":
                if u_r is None:
                    u_r = model.solve_state(q_r, use_cached_operators=use_cached_operators)

                _u_r = reductor.reconstruct(u_r, basis="state_basis")
                u = self.FOM.solve_state(q, use_cached_operators=use_cached_operators)

                diff = u - _u_r
                err = np.sqrt(self.FOM.products["bochner_prod_V"].apply2(diff, diff))[0, 0]
                out["err_u"] = float(err)

                norm_u = np.sqrt(self.FOM.products["bochner_prod_V"].apply2(u, u))[0, 0]
                out["rel_err_u"] = float(err / norm_u)

                if self.logger:
                    self.logger.debug(f"Actual err_u = {out['err_u']:3.4e}")
                    self.logger.debug(f"Actual rel_err_u = {out['rel_err_u']:3.4e}")
                continue

            if rq == "p":
                if p_r is None:
                    p_r = model.solve_adjoint(q_r, u_r, use_cached_operators=use_cached_operators)

                _p_r = reductor.reconstruct(p_r, basis="state_basis")
                p = self.FOM.solve_adjoint(q, u, use_cached_operators=use_cached_operators)

                diff = p - _p_r
                err = np.sqrt(self.FOM.products["bochner_prod_V"].apply2(diff, diff))[0, 0]
                out["err_p"] = float(err)

                norm_p = np.sqrt(self.FOM.products["bochner_prod_V"].apply2(p, p))[0, 0]
                out["rel_err_p"] = float(err / norm_p)

                if self.logger:
                    self.logger.debug(f"Actual err_p = {out['err_p']:3.4e}")
                    self.logger.debug(f"Actual rel_err_p = {out['rel_err_p']:3.4e}")
                continue

            if rq == "lin_u":
                if lin_u_r is None:
                    lin_u_r = model.solve_linearized_state(q_r, d_r, u_r, use_cached_operators=use_cached_operators)

                _lin_u_r = reductor.reconstruct(lin_u_r, basis="state_basis")
                lin_u = self.FOM.solve_linearized_state(q, d, u, use_cached_operators=use_cached_operators)

                diff = lin_u - _lin_u_r
                err = np.sqrt(self.FOM.products["bochner_prod_V"].apply2(diff, diff))[0, 0]
                out["err_lin_u"] = float(err)

                norm_lin_u = np.sqrt(self.FOM.products["bochner_prod_V"].apply2(lin_u, lin_u))[0, 0]
                out["rel_err_lin_u"] = float(err / norm_lin_u)

                if self.logger:
                    self.logger.debug(f"Actual err_lin_u = {out['err_lin_u']:3.4e}")
                    self.logger.debug(f"Actual rel_err_lin_u = {out['rel_err_lin_u']:3.4e}")
                continue

            if rq == "lin_p":
                if lin_p_r is None:
                    lin_p_r = model.solve_linearized_adjoint(q_r, u_r, lin_u_r, use_cached_operators=use_cached_operators)

                _lin_p_r = reductor.reconstruct(lin_p_r, basis="state_basis")
                lin_p = self.FOM.solve_linearized_adjoint(q, u, lin_u, use_cached_operators=use_cached_operators)

                diff = lin_p - _lin_p_r
                err = np.sqrt(self.FOM.products["bochner_prod_V"].apply2(diff, diff))[0, 0]
                out["err_lin_p"] = float(err)

                norm_lin_p = np.sqrt(self.FOM.products["bochner_prod_V"].apply2(lin_p, lin_p))[0, 0]
                out["rel_err_lin_p"] = float(err / norm_lin_p)

                if self.logger:
                    self.logger.debug(f"Actual err_lin_p = {out['err_lin_p']:3.4e}")
                    self.logger.debug(f"Actual rel_err_lin_p = {out['rel_err_lin_p']:3.4e}")
                continue

            if rq == "J":
                if J_r_ is None:
                    J_r_ = model.objective(u_r, q_r)

                J = self.FOM.objective(u, q)
                err = np.abs(J - J_r_)
                out["err_J"] = float(err)
                out["rel_err_J"] = float(err / np.abs(J))

                if self.logger:
                    self.logger.debug(f"Actual err_J = {out['err_J']:3.4e}")
                    self.logger.debug(f"Actual rel_err_J = {out['rel_err_J']:3.4e}")
                continue

            if rq == "nabla_J":
                if nabla_J_r_ is None:
                    nabla_J_r_ = model.gradient(u_r, p_r, q_r, use_cached_operators=use_cached_operators)

                _nabla_J_r = reductor.reconstruct(nabla_J_r_, basis="parameter_basis")
                nabla_J = self.FOM.gradient(u, p, q, use_cached_operators=use_cached_operators)

                diff = nabla_J - _nabla_J_r
                if self.FOM.q_time_dep:
                    err = np.sqrt(self.FOM.products["bochner_prod_Q"].apply2(diff, diff))[0, 0]
                    norm = np.sqrt(self.FOM.products["bochner_prod_Q"].apply2(nabla_J, nabla_J))[0, 0]
                else:
                    err = np.sqrt(self.FOM.products["prod_Q"].apply2(diff, diff))[0, 0]
                    norm = np.sqrt(self.FOM.products["prod_Q"].apply2(nabla_J, nabla_J))[0, 0]

                out["err_nabla_J"] = float(err)
                out["rel_err_nabla_J"] = float(err / norm)

                if self.logger:
                    self.logger.debug(f"Actual err_nabla_J = {out['err_nabla_J']:3.4e}")
                    self.logger.debug(f"Actual rel_err_nabla_J = {out['rel_err_nabla_J']:3.4e}")
                continue

            if rq == "lin_J":
                if lin_J_r_ is None:
                    lin_J_r_ = model.linearized_objective(
                        q_r, d_r, u_r, lin_u_r, 0.0, use_cached_operators=use_cached_operators
                    )

                lin_J = self.FOM.linearized_objective(q, d, u, lin_u, 0.0, use_cached_operators=use_cached_operators)
                err = np.abs(lin_J - lin_J_r_)
                out["err_lin_J"] = float(err)
                out["rel_err_lin_J"] = float(err / np.abs(lin_J))

                if self.logger:
                    self.logger.debug(f"Actual err_lin_J = {out['err_lin_J']:3.4e}")
                    self.logger.debug(f"Actual rel_err_lin_J = {out['rel_err_lin_J']:3.4e}")
                continue

            if rq == "nabla_lin_J":
                if nabla_lin_J_r_ is None:
                    nabla_lin_J_r_ = model.linearized_gradient(
                        q_r, d_r, u_r, lin_p_r, 0.0, use_cached_operators=use_cached_operators
                    )

                _nabla_lin_J_r = reductor.reconstruct(nabla_lin_J_r_, basis="parameter_basis")
                nabla_lin_J = self.FOM.linearized_gradient(q, d, u, lin_u, 0.0, use_cached_operators=use_cached_operators)

                diff = nabla_lin_J - _nabla_lin_J_r
                if self.FOM.q_time_dep:
                    err = np.sqrt(self.FOM.products["bochner_prod_Q"].apply2(diff, diff))[0, 0]
                    norm = np.sqrt(self.FOM.products["bochner_prod_Q"].apply2(nabla_lin_J, nabla_lin_J))[0, 0]
                else:
                    err = np.sqrt(self.FOM.products["prod_Q"].apply2(diff, diff))[0, 0]
                    norm = np.sqrt(self.FOM.products["prod_Q"].apply2(nabla_lin_J, nabla_lin_J))[0, 0]

                out["err_nabla_lin_J"] = float(err)
                out["rel_err_nabla_lin_J"] = float(err / norm)

                if self.logger:
                    self.logger.debug(f"Actual err_nabla_lin_J = {out['err_nabla_lin_J']:3.4e}")
                    self.logger.debug(f"Actual rel_err_nabla_lin_J = {out['rel_err_nabla_lin_J']:3.4e}")
                continue

            raise ValueError(f"Unexpected required quantity: {rq}")

        return out
