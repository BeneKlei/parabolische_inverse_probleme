from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
from pymor.algorithms.hapod import inc_vectorarray_hapod
from pymor.core.base import BasicObject
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray

from RBInvParam.model import InstationaryModelIP
from RBInvParam.utils.logger import get_default_logger


class SnapshotPreprocessor(BasicObject):
    def __init__(
        self,
        FOM: InstationaryModelIP,
        active_bases: List[str],
        use_adjoint_space: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        
        logging.basicConfig()

        if logger:
            self._logger = logger
        else:
            self._logger = get_default_logger(self.__class__.__name__)
            self._logger.setLevel(logging.DEBUG)
        self.logger.debug(f"Setting up {self.__class__.__name__}")

        if not isinstance(FOM, InstationaryModelIP):
            raise TypeError("FOM must be an InstationaryModelIP")

        self.FOM = FOM
        self.active_bases = active_bases
        self.use_adjoint_space = use_adjoint_space

        if self.use_adjoint_space and "adjoint_basis" in self.active_bases:
            raise ValueError("When use_adjoint_space=True, do not include 'adjoint_basis' in active_bases")

        self.krylov_directions = self.FOM.Q.empty()
        self.krylov_sensitivities = self.FOM.V.empty()

    # --------------------------------------------------
    # utils
    # --------------------------------------------------
        
    def make_empty_snapshots_dict(self) -> Dict[str, VectorArray]:
        space_for = {
            "parameter_basis": self.FOM.Q,
            "state_basis": self.FOM.V,
            "adjoint_basis": self.FOM.V,
        }
        return {b: space_for[b].empty() for b in self.active_bases}
    
    # --------------------------------------------------
    # Helpers for common intermediates
    # --------------------------------------------------
    
    def _compute_krylov(
        self,
        cfg: Dict,
        *,
        q: VectorArray,
        u: VectorArray,
        use_cached_operators: bool,
        nabla_J: VectorArray,
    ) -> None:
        """
        Computes krylov directions and sensitivities and appends to
        self.krylov_directions / self.krylov_sensitivities.
        """
        n = int(cfg["n"])
        initial_direction = cfg.get("initial_direction", "gradient")

        self.logger.debug("Include krylov directions, with n = %d", n)

        if initial_direction == "gradient":
            krylov_direction = nabla_J
        elif initial_direction == "ones":
            krylov_direction = self.FOM.Q.ones()
        else:
            raise ValueError(f"Unknown initial_direction: {initial_direction!r}")

        self.krylov_directions.append(krylov_direction)

        for _j in range(n):
            lin_u = self.FOM.solve_linearized_state(
                q=q,
                d=krylov_direction,
                u=u,
                use_cached_operators=use_cached_operators,
            )
            self.krylov_sensitivities.append(lin_u)

            z = self.FOM.solve_second_adjoint(
                q=q,
                lin_u=lin_u,
                use_cached_operators=use_cached_operators,
            )

            krylov_direction = self.FOM.gauss_newton_hessian(
                u=u,
                z=z,
                q=q,
                use_cached_operators=use_cached_operators,
            )
            self.krylov_directions.append(krylov_direction)

        # match your original “extra” sensitivity at the end
        lin_u = self.FOM.solve_linearized_state(
            q=q,
            d=krylov_direction,
            u=u,
            use_cached_operators=use_cached_operators,
        )
        self.krylov_sensitivities.append(lin_u)
    
    def _compute_linearized_states(
        self,
        q: VectorArray,
        u: VectorArray,
        use_cached_operators: bool,
    ) -> tuple[VectorArray, VectorArray]:
        
        # direction = ones; consider making this configurable or passing in
        direction = self.FOM.Q.make_array(np.ones(self.FOM.Q.dim))
        lin_u = self.FOM.solve_linearized_state(q, direction, u, use_cached_operators=use_cached_operators)
        lin_p = self.FOM.solve_linearized_adjoint(q, u, lin_u, use_cached_operators=use_cached_operators)
        return lin_u, lin_p
    
    def _additional_parameter_snapshots(
        self,
        cfg: Dict,
        q: Optional[VectorArray] = None,
        u: Optional[VectorArray] = None,
        nabla_J: Optional[VectorArray] = None,
        nabla_lin_J: Optional[VectorArray] = None,
        time_steps_nabla_J: Optional[VectorArray] = None,
        time_steps_nabla_lin_J: Optional[VectorArray] = None,
        use_cached_operators: bool = False,
    ) -> VectorArray:
            
        parameter_snapshots = self.FOM.Q.empty()

        if cfg.get("include_each_nabla_J_time_step", False) and not self.FOM.q_time_dep:
            if time_steps_nabla_J is None:
                raise ValueError("time_steps_nabla_J required")
            self.logger.debug("Include gradients per time step as snapshots")
            parameter_snapshots.append(time_steps_nabla_J)

        if cfg.get("include_each_nabla_lin_J_time_step", False) and not self.FOM.q_time_dep:
            if time_steps_nabla_lin_J is None:
                raise ValueError("time_steps_nabla_lin_J required")
            self.logger.debug("Include linearized gradients per time step as snapshots")
            parameter_snapshots.append(time_steps_nabla_lin_J)

        if cfg.get("include_lin_grad", False):
            if nabla_lin_J is None:
                raise ValueError("nabla_lin_J required")
            self.logger.debug("Include nabla_lin_J")
            parameter_snapshots.append(nabla_lin_J)

        if cfg.get("include_krylov_directions", False):
            raise NotImplementedError
        
        if cfg.get("include_q_exact", False):
            parameter_snapshots.append(
                self.FOM.Q.make_array(self.FOM.setup['q_exact'])
            )


        return parameter_snapshots

    def _additional_state_snapshots(
        self,
        cfg: Dict,
        lin_u: Optional[VectorArray] = None,
        lin_p: Optional[VectorArray] = None,
    ) -> VectorArray:

        state_snapshots = self.FOM.V.empty()

        if cfg.get("include_lin_states", False):
            if lin_u is None or lin_p is None:
                raise ValueError("lin_u and lin_p required")
            self.logger.debug("Include linearized states")
            state_snapshots.append(lin_u)
            state_snapshots.append(lin_p)

        if cfg.get("include_krylov_sensitivites", False):
            raise NotImplementedError

        return state_snapshots
    
    # --------------------------------------------------
    # Snapshot assembly
    # --------------------------------------------------
    
    def get_additional_snapshots(
        self,
        config: Dict[str, Dict],
        bases: List[str],
        q: Optional[VectorArray] = None,
        u: Optional[VectorArray] = None,
        nabla_J: Optional[VectorArray] = None,
        time_steps_nabla_J: Optional[VectorArray] = None,
        use_cached_operators: bool = False,
    ) -> Dict[str, VectorArray]:
        if not set(bases).issubset(self.active_bases):
            raise ValueError("Requested bases must be subset of active_bases")

        additional_snapshots = self.make_empty_snapshots_dict()

        # Read configs safely with defaults
        param_cfg = (config.get("parameter_basis") or {}).get("additional_snapshots", {}) or {}
        state_cfg = (config.get("state_basis") or {}).get("additional_snapshots", {}) or {}
    
        include_lin_grad = param_cfg.get("include_lin_grad", False)
        include_each_lin = param_cfg.get("include_each_nabla_lin_J_time_step", False)
        include_krylov_directions = param_cfg.get("include_krylov_directions", False)
        include_q_exact = param_cfg.get("include_q_exact", False)

        include_lin_states = state_cfg.get("include_lin_states", False)
        include_krylov_sensitivites = state_cfg.get("include_krylov_sensitivites", False)

        need_linearized = include_lin_states or include_lin_grad or include_each_lin

        lin_u = lin_p = None
        if need_linearized:
            if q is None or u is None or use_cached_operators is None:
                raise ValueError("q, u, use_cached_operators required for linearized quantities")
            lin_u, lin_p = self._compute_linearized_quantities(q, u, use_cached_operators)

        nabla_lin_J = time_steps_nabla_lin_J = None
        if include_lin_grad or include_each_lin:
            if q is None or u is None or nabla_J is None or lin_p is None or use_cached_operators is None:
                raise ValueError("q, u, nabla_J, lin_p, use_cached_operators required for linearized gradient")
            nabla_lin_J, time_steps_nabla_lin_J = self.FOM.linearized_gradient(
                q,
                nabla_J,
                u,
                lin_p,
                alpha=0,
                use_cached_operators=use_cached_operators,
                return_per_time_step=True,
            )

        if include_krylov_directions or include_krylov_sensitivites:
            raise NotImplementedError


        for basis in bases:
            if basis == "parameter_basis":
                additional_snapshots[basis] = self._additional_parameter_snapshots(
                    param_cfg,
                    q=q,
                    u=u,
                    nabla_J=nabla_J,
                    nabla_lin_J=nabla_lin_J,
                    time_steps_nabla_J=time_steps_nabla_J,
                    time_steps_nabla_lin_J=time_steps_nabla_lin_J,
                    use_cached_operators=use_cached_operators,
                )
            elif basis == "state_basis":
                additional_snapshots[basis] = self._additional_state_snapshots(state_cfg, lin_u=lin_u, lin_p=lin_p)
            elif basis == "adjoint_basis":
                # TODO if needed
                pass
            else:
                raise ValueError(f"Unknown basis: {basis}")

        return additional_snapshots

    def get_snapshots(
        self,
        config: Dict[str, Dict],
        bases: List[str],
        q: Optional[VectorArray] = None,
        u: Optional[VectorArray] = None,
        p: Optional[VectorArray] = None,
        nabla_J: Optional[VectorArray] = None,
        time_steps_nabla_J: Optional[VectorArray] = None,
        add_additional_snapshots: bool = True,
        use_cached_operators: bool = False,
        ) -> Dict[str, VectorArray]:
        
        if not set(bases).issubset(self.active_bases):
            raise ValueError("Requested bases must be subset of active_bases")

        snapshots = self.make_empty_snapshots_dict()
        additional = None

        if add_additional_snapshots:
            additional = self.get_additional_snapshots(
                config=config,
                bases=bases,
                q=q,
                u=u,
                nabla_J=nabla_J,
                time_steps_nabla_J=time_steps_nabla_J,
                use_cached_operators=use_cached_operators,
            )

        def _append_additional(basis: str) -> None:
            if additional is None:
                return
            extra = additional.get(basis)
            if extra is None:
                return
            # extra is a VectorArray (possibly empty)
            snapshots[basis].append(extra)

        for basis in bases:
            self.logger.debug("Extending '%s' snapshots", basis)

            if basis == "parameter_basis":
                if nabla_J is None:
                    raise ValueError("nabla_J is required for parameter_basis snapshots")
                if nabla_J not in self.FOM.Q:
                    raise ValueError("nabla_J must live in FOM.Q")

                snapshots["parameter_basis"].append(nabla_J)
                _append_additional("parameter_basis")

            elif basis == "state_basis":
                if u is None:
                    raise ValueError("u is required for state_basis snapshots")
                if u not in self.FOM.V:
                    raise ValueError("u must live in FOM.V")

                snapshots["state_basis"].append(u)

                if not self.use_adjoint_space:
                    if p is None:
                        raise ValueError("p is required for state_basis snapshots when use_adjoint_space=False")
                    if p not in self.FOM.V:
                        raise ValueError("p must live in FOM.V")
                    snapshots["state_basis"].append(p)

                _append_additional("state_basis")

            elif basis == "adjoint_basis":
                if not self.use_adjoint_space:
                    raise ValueError("adjoint_basis snapshots only make sense when use_adjoint_space=True")

                if p is None:
                    raise ValueError("p is required for adjoint_basis snapshots")
                if p not in self.FOM.V:
                    raise ValueError("p must live in FOM.V")

                snapshots["adjoint_basis"].append(p)
                _append_additional("adjoint_basis")

            else:
                raise ValueError(f"Unknown basis: {basis}")

        return snapshots

    # --------------------------------------------------
    # Transforming Snapshots
    # --------------------------------------------------

    def _HaPOD(
        self,
        snapshots: VectorArray,
        product: Operator,
        eps: float,
        omega: float,
        steps: Optional[int] = None,
    ) -> Tuple[VectorArray, List[float], int]:
        if not isinstance(snapshots, VectorArray):
            raise TypeError("snapshots must be a VectorArray")
        if product.source != product.range or product.source != snapshots.space:
            raise ValueError("product must map snapshots.space -> snapshots.space")

        n = len(snapshots)
        use_steps = steps if steps is not None else max(1, n // 2)

        return inc_vectorarray_hapod(
            steps=use_steps,
            U=snapshots,
            eps=eps,
            omega=omega,
            product=product,
        )

    def preprocess(
        self,
        snapshots: VectorArray,
        product: Operator,
        config: Dict,
    ) -> VectorArray:
        if not isinstance(snapshots, VectorArray):
            raise TypeError("snapshots must be a VectorArray")
        if product.source != product.range or product.source != snapshots.space:
            raise ValueError("product must map snapshots.space -> snapshots.space")

        self.logger.debug("Starting snapshot preprocessing")

        if not config:
            return snapshots

        # --- Select every n-th snapshot (always keep first and last)
        every_n = config.get("every_n")
        if every_n is not None:
            if not isinstance(every_n, int) or every_n <= 0:
                raise ValueError("config['every_n'] must be a positive integer")

            N = len(snapshots)
            if N > 0:
                indices = [i for i in range(N) if i == 0 or i == N - 1 or i % every_n == 0]
                self.logger.debug(
                    "  Applying 'every_n=%d': keeping %d of %d snapshots",
                    every_n,
                    len(indices),
                    N,
                )
                snapshots = snapshots[indices]

        # --- Normalize
        if config.get("normalize", False):
            self.logger.debug("  Applying 'normalize'")
            norms = snapshots.norm(product)

            # norms is a numpy array-like; avoid division by ~0
            norms = np.asarray(norms, dtype=float)
            safe = np.where(norms <= 1e-16, 1.0, norms)
            snapshots.scal(1.0 / safe)

        # --- HaPOD (keep optional)
        hapod_cfg: Optional[Dict] = config.get("HaPOD")  # None/False disables
        if hapod_cfg:
            eps = hapod_cfg["eps"]
            omega = hapod_cfg["omega"]
            steps = hapod_cfg.get("steps")  # optional if you support it

            self.logger.debug("  Applying 'HaPOD' (eps=%s, omega=%s, steps=%s)", eps, omega, steps)

            snapshots, svals, snap_count = self._HaPOD(
                snapshots=snapshots,
                product=product,
                eps=eps,
                omega=omega,
                steps=steps,
            )

            self.logger.debug(
                "  HaPOD returned %d modes (snap_count=%s).",
                len(snapshots),
                snap_count,
            )
            # Keep svals logging, but don’t spam huge arrays
            if svals is not None:
                self.logger.debug("  Singular values (first 10): %s", list(svals[:10]))

        return snapshots
