from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from RBInvParam.schemas.tcc_evaluator import *

class TCCEvaluator:
    """
    Tangential cone condition analyzer for one or multiple models.
    """

    def __init__(
        self,
        config: Optional[TCCEvaluatorConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or TCCEvaluatorConfig.defaults()
        self.config.validate()

        if logger is None:
            logging.basicConfig()
            logger = logging.getLogger(self.__class__.__name__)
            logger.setLevel(logging.INFO)
        self.logger = logger

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run_single_model(
        self,
        model: Any,
        q=None,
        model_name: Optional[str] = None,
        config: Optional[TCCEvaluatorConfig] = None,
    ) -> Dict[str, Any]:
        cfg = config or self.config
        cfg.validate()

        if q is None:
            q = model.q_circ

        if model_name is None:
            model_name = getattr(model, "name", model.__class__.__name__)

        rng = np.random.default_rng(cfg.seed)
        q_np = q.to_numpy()
        flat_size = q_np.size

        grid_shape = self._get_grid_shape(model)
        n_samples = min(cfg.max_h, flat_size)
        sampled_indices = rng.choice(flat_size, size=n_samples, replace=False)

        u_q = model.solve_state(q)
        Cu_q = model.C.apply(u_q)
        norm_Cu_q = self._norm_C(model, Cu_q)

        direction_data = self._prepare_direction_data(
            model=model,
            q=q,
            mode=cfg.perturbation_mode,
        )

        all_data = []

        self.logger.info(
            "Starting TCC analysis for model='%s', mode='%s'",
            model_name,
            cfg.perturbation_mode.value,
        )

        for amplitude in cfg.amplitudes:
            self.logger.info("=" * 100)
            self.logger.info("[%s] amplitude = %.4e", model_name, amplitude)

            rows = []

            for sample_id, idx in enumerate(sampled_indices):
                d = self._build_perturbation(
                    model=model,
                    q_np=q_np,
                    amplitude=amplitude,
                    idx=int(idx),
                    mode=cfg.perturbation_mode,
                    direction_data=direction_data,
                )

                values = self._evaluate_tcc(
                    model=model,
                    q=q,
                    d=d,
                    u_q=u_q,
                    Cu_q=Cu_q,
                    norm_Cu_q=norm_Cu_q,
                )

                row, col = np.unravel_index(idx, grid_shape)

                result_row = {
                    "sample": int(sample_id),
                    "flat_index": int(idx),
                    "row": int(row),
                    "col": int(col),
                    "amplitude": float(amplitude),
                    **values,
                }
                rows.append(result_row)

                if cfg.verbose_logging:
                    self.logger.info(
                        "[%s] sample=%2d | node=(%2d,%2d) | ||d||=%.3e | "
                        "||rem||=%.3e | ||diff||=%.3e | ||lin||=%.3e | TCC=%.3e",
                        model_name,
                        sample_id,
                        row,
                        col,
                        result_row["norm_d"],
                        result_row["norm_rem"],
                        result_row["norm_diff"],
                        result_row["norm_lin"],
                        result_row["c_tc"],
                    )

            c_vals = np.array([r["c_tc"] for r in rows], dtype=float)

            amp_result = {
                "model_name": model_name,
                "perturbation_mode": cfg.perturbation_mode.value,
                "amplitude": float(amplitude),
                "rows": rows,
                "mean": float(np.nanmean(c_vals)),
                "median": float(np.nanmedian(c_vals)),
                "min": float(np.nanmin(c_vals)),
                "max": float(np.nanmax(c_vals)),
            }
            all_data.append(amp_result)

            if cfg.verbose_logging:
                self.logger.info(
                    "[%s] amplitude=%.4e | mean=%.3e | median=%.3e | min=%.3e | max=%.3e",
                    model_name,
                    amp_result["amplitude"],
                    amp_result["mean"],
                    amp_result["median"],
                    amp_result["min"],
                    amp_result["max"],
                )
            else:
                self.logger.info(
                    "[%s] amplitude=%.4e | mean_TCC=%.3e",
                    model_name,
                    amp_result["amplitude"],
                    amp_result["mean"],
                )

        return {
            "model_name": model_name,
            "config": {
                "amplitudes": list(cfg.amplitudes),
                "max_h": cfg.max_h,
                "seed": cfg.seed,
                "perturbation_mode": cfg.perturbation_mode.value,
                "create_pdf": cfg.create_pdf,
                "pdf_filename": str(cfg.pdf_filename),
            },
            "results": all_data,
        }

    def run_multiple_models(
        self,
        models: Dict[str, Any],
        q_map: Optional[Dict[str, Any]] = None,
        config: Optional[TCCEvaluatorConfig] = None,
    ) -> Dict[str, Any]:
        cfg = config or self.config
        cfg.validate()

        if q_map is None:
            q_map = {}

        comparison_data = {
            "config": {
                "amplitudes": list(cfg.amplitudes),
                "max_h": cfg.max_h,
                "seed": cfg.seed,
                "perturbation_mode": cfg.perturbation_mode.value,
                "create_pdf": cfg.create_pdf,
                "pdf_filename": str(cfg.pdf_filename),
            },
            "models": {},
        }

        for model_name, model in models.items():
            result = self.run_single_model(
                model=model,
                q=q_map.get(model_name, None),
                model_name=model_name,
                config=cfg,
            )
            comparison_data["models"][model_name] = result

        if cfg.create_pdf:
            self.create_comparison_pdf_report(
                comparison_data=comparison_data,
                pdf_filename=cfg.pdf_filename,
            )
            self.logger.info("Saved TCC comparison PDF to: %s", cfg.pdf_filename)

        return comparison_data

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_grid_shape(self, model: Any) -> tuple[int, int]:
        shape_cfg = model.setup["param_grid_resolution"][1:]
        return (shape_cfg[0] + 1, shape_cfg[1] + 1)

    def _norm_Q(self, model: Any, d) -> float:
        return float(np.sqrt(model.products["prod_Q"].apply2(d, d))[0, 0])

    def _norm_C(self, model: Any, v) -> float:
        return float(np.sqrt(model.products["bochner_prod_C"].apply2(v, v))[0, 0])

    def _prepare_direction_data(
        self,
        model: Any,
        q,
        mode: TCCPerturbationType,
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {}

        if mode == TCCPerturbationType.GRADIENT_DIRECTION:
            nabla_J = model.compute_gradient(q)
            grad_norm = self._norm_Q(model, nabla_J)
            if grad_norm <= 0:
                raise ValueError("Gradient norm is zero; cannot use gradient_direction.")
            data["direction"] = (1.0 / grad_norm) * nabla_J

        elif mode == TCCPerturbationType.Q_EXACT_DIRECTION:
            q_exact = model.Q.make_array(model.setup["q_exact"])
            direction = q_exact - q
            dir_norm = self._norm_Q(model, direction)
            if dir_norm <= 0:
                raise ValueError("q_exact - q has zero norm; cannot use q_exact_direction.")
            data["direction"] = (1.0 / dir_norm) * direction

        elif mode == TCCPerturbationType.LOCAL_BASIS:
            pass

        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")

        return data

    def _build_perturbation(
        self,
        model: Any,
        q_np: np.ndarray,
        amplitude: float,
        idx: int,
        mode: TCCPerturbationType,
        direction_data: Dict[str, Any],
    ):
        if mode in {
            TCCPerturbationType.GRADIENT_DIRECTION,
            TCCPerturbationType.Q_EXACT_DIRECTION,
        }:
            return amplitude * direction_data["direction"]

        if mode == TCCPerturbationType.LOCAL_BASIS:
            d_np = np.zeros_like(q_np)
            np.put(d_np, idx, amplitude)
            return model.Q.make_array(d_np)

        raise NotImplementedError(f"Unsupported mode: {mode}")

    def _evaluate_tcc(
        self,
        model: Any,
        q,
        d,
        u_q=None,
        Cu_q=None,
        norm_Cu_q: Optional[float] = None,
    ) -> Dict[str, float]:
        if u_q is None:
            u_q = model.solve_state(q)

        if Cu_q is None:
            Cu_q = model.C.apply(u_q)

        if norm_Cu_q is None:
            norm_Cu_q = self._norm_C(model, Cu_q)

        u_prime_qd = model.solve_linearized_state(q, d, u_q)
        Cu_prime_qd = model.C.apply(u_prime_qd)

        u_qd = model.solve_state(q + d)
        Cu_qd = model.C.apply(u_qd)

        remainder = Cu_qd - Cu_q - Cu_prime_qd
        increment = Cu_qd - Cu_q

        norm_Cu_qd = self._norm_C(model, Cu_qd)
        norm_diff = self._norm_C(model, increment)
        norm_lin = self._norm_C(model, Cu_prime_qd)
        norm_rem = self._norm_C(model, remainder)
        norm_d = self._norm_Q(model, d)

        c_tc = norm_rem / norm_diff if norm_diff > 0 else np.nan

        return {
            "norm_d": float(norm_d),
            "norm_Cu_q": float(norm_Cu_q),
            "norm_Cu_qd": float(norm_Cu_qd),
            "norm_diff": float(norm_diff),
            "norm_lin": float(norm_lin),
            "norm_rem": float(norm_rem),
            "c_tc": float(c_tc) if np.isfinite(c_tc) else np.nan,
        }

    def create_comparison_pdf_report(
        self,
        comparison_data: Dict[str, Any],
        pdf_filename: Union[str, Path],
    ) -> None:
        raise NotImplementedError("Implement your plotting/reporting here.")