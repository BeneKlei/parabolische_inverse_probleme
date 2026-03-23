import numpy as np
import logging
from pathlib import Path
from datetime import datetime

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"


from pymor.basic import *
from pymor.core.defaults import set_defaults, get_defaults
from pymor.algorithms.genericsolvers import solver_options

import RBInvParam.problems.hyperelasticity.hyperelasticity_model as hm
import RBInvParam.problems.shared.material_model as mm

#import material_model as mm
#import hyperelasticity_model as hm

from RBInvParam.optimizer.optimizer import QrVrROMOptimizer, LoggerErrorChoice
from RBInvParam.utils.io import save_dict_to_pkl
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.problems.hyperelasticity.build import build_HyperElasticityModelIP

from RBInvParam.error_estimators.state_error_estimators import StateErrorEstimatorType
from RBInvParam.error_estimators.adjoint_error_estimators import AdjointErrorEstimatorType
from RBInvParam.error_estimators.objective_error_estimators import ObjectiveErrorEstimatorType
from RBInvParam.trust_region import TRType
from RBInvParam.schemas.reductor import LinearizationMethod

from RBInvParam.timestepping import TimeStepperType

from RBInvParam.utils.create_q_exact import *

#########################################################################################''

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = Path('./dumps') / (timestamp + '_TR_IRGNM')
os.mkdir(save_path)
logfile_path= save_path / 'TR_IRGNM.log'

logger = get_default_logger(logger_name='TR_IRGNM',
                            logfile_path=logfile_path, 
                            use_timestemp=False)
logger.setLevel(logging.DEBUG)

#########################################################################################''
set_log_levels({
    #'pymor.operators.constructions.LincombOperator' : 'ERROR',
    #'pymor.operators.constructions.AdjointOperator' : 'ERROR',
    #'pymor.algorithms.genericsolvers.lgmres' : 'ERROR',
    #'pymor.algorithms' : 'ERROR'
})

set_defaults({
    #'pymor.algorithms.genericsolvers.solver_options.lgmres_tol' : 1e-12,
    #'pymor.algorithms.genericsolvers.solver_options.lgmres_maxiter' : int(1e3),
    #'pymor.algorithms.newton.newton.maxiter' : 1e3,
    #'pymor.algorithms.newton.newton.atol' : 1e-4,
    #'pymor.algorithms.newton.newton.atol' : 1e-5,
})



#########################################################################################''

# np.set_printoptions(linewidth=np.inf) 
# np.set_printoptions(threshold=np.inf)  # force full print

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from pymor.vectorarrays.interface import VectorArray
from pymor.operators.interface import Operator
from typing import Dict, Tuple

class NormDiffModel():
    
    def __init__(self,
                 q_circ: VectorArray,
                 q_exact: VectorArray,
                 Q : Operator,
                 C : Operator,
                 products : Dict,
                 setup : Dict,
                 nt : int):
        
        self.q_circ = q_circ
        self.q_exact = q_exact
        self.Q = Q
        self.C = C
        self.products = products
        self.setup = setup
        self.nt = nt
        
    def solve_state(self, 
                    q: VectorArray,
                    use_cached_operators: bool = False,
                    return_higher_orders: bool = False) -> VectorArray | Tuple[VectorArray,VectorArray]:

        _q = (q - self.q_circ).to_numpy()
        _q = 0.5 * _q**2

        u = self.products['prod_V'].source.zeros(self.nt + 1)
        # fill the first vector with _q data
        # exact line depends on your dealii wrapper
        for i in range(self.nt+1):
            u.vectors[i].impl[:_q.shape[1]] = _q[0]
        return u
        

    def solve_linearized_state(self,
                               q: VectorArray,
                               d: VectorArray,
                               u: VectorArray,
                               use_cached_operators: bool = False,
                               return_higher_orders: bool = False) -> VectorArray | Tuple[VectorArray,VectorArray]:

        _q = (q - self.q_circ).to_numpy()
        _d = d.to_numpy()

        lin = _q * _d   # derivative of 0.5 * (q - q_circ)^2 in direction d

        out = self.products['prod_V'].source.zeros(self.nt + 1)
        for i in range(self.nt + 1):
            out.vectors[i].impl[:lin.shape[1]] = lin[0]
        return out
                  
def run_tcc_analysis(
    FOM,
    q=None,
    amplitudes=(1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10),
    max_h=1,
    seed=0,
    use_gradient_direction=True,
    pdf_filename="tcc_analysis.pdf",
):
    """
    Run TCC analysis over several amplitudes and save all plots into one PDF.

    TCC:
        c_tc = ||u(q+d) - u(q) - u'(q)d|| / ||u(q+d) - u(q)||

    Parameters
    ----------
    FOM : object
        Full-order model with the required methods/products.
    q : parameter array, optional
        Reference parameter. If None, uses FOM.q_circ.
    amplitudes : iterable of float
        Perturbation amplitudes.
    max_h : int
        Number of sampled perturbation nodes.
    seed : int
        RNG seed.
    use_gradient_direction : bool
        If True, use d = amplitude * grad J(q).
        If False, use localized perturbation at sampled node.
    pdf_filename : str
        Output PDF filename.

    Returns
    -------
    all_data : list of dict
        Per-amplitude results.
    """

    rng = np.random.default_rng(seed)

    if q is None:
        q = FOM.q_circ

    q_np = q.to_numpy()
    shape = FOM.setup["param_grid_resolution"][1:]
    shape = (shape[0] + 1, shape[1] + 1)
    flat_size = q_np.size

    n_samples = min(max_h, flat_size)
    indices = rng.choice(flat_size, size=n_samples, replace=False)

    # Reference state
    u_q = FOM.solve_state(q)
    Cu_q = FOM.C.apply(u_q)
    norm_Cu_q_ref = np.sqrt(FOM.products["bochner_prod_C"].apply2(Cu_q, Cu_q))[0, 0]


    all_data = []

    for amplitude in amplitudes:
        print("==========================================")
        print(f"amplitude = {amplitude:3.4e}")

        rows = []

        for k, idx in enumerate(indices):
            # Build perturbation
            if use_gradient_direction:
                nabla_J = FOM.compute_gradient(q)
                d_np = nabla_J.to_numpy()
            else:
                d_np = np.zeros_like(q_np)
                
                # d_grid = d_np.reshape(shape)
                # d_grid[3:7, 3:7] = amplitude
                # d_np = d_grid.reshape(q_np.shape)
                
                np.put(d_np, idx, amplitude)

            # d = FOM.Q.make_array(d_np)
            # d = amplitude / np.sqrt(FOM.products["prod_Q"].apply2(d, d))[0, 0] * d

            d = FOM.Q.make_array(FOM.setup['q_exact']) - q
            #d = amplitude * d
            d = amplitude  / np.sqrt(FOM.products["prod_Q"].apply2(d, d))[0, 0] * d
            

            # Solve states
            u_prime_qd = FOM.solve_linearized_state(q, d, u_q)
            Cu_prime_qd = FOM.C.apply(u_prime_qd)

            u_qd = FOM.solve_state(q + d)
            Cu_qd = FOM.C.apply(u_qd)

            # Quantities
            a = Cu_qd - Cu_q - Cu_prime_qd   # remainder
            b = Cu_qd - Cu_q                 # full increment

            norm_Cu_q = norm_Cu_q_ref
            norm_Cu_qd = np.sqrt(FOM.products["bochner_prod_C"].apply2(Cu_qd, Cu_qd))[0, 0]
            norm_diff = np.sqrt(FOM.products["bochner_prod_C"].apply2(b, b))[0, 0]
            norm_lin = np.sqrt(FOM.products["bochner_prod_C"].apply2(Cu_prime_qd, Cu_prime_qd))[0, 0]
            norm_rem = np.sqrt(FOM.products["bochner_prod_C"].apply2(a, a))[0, 0]
            norm_d = np.sqrt(FOM.products["prod_Q"].apply2(d, d))[0, 0]

            c_tc = norm_rem / norm_diff if norm_diff != 0 else np.nan
            row, col = np.unravel_index(idx, shape)

            print(
                f"sample {k:2d} | node=({row:2d},{col:2d}) | "
                f"||d||={norm_d:.3e} | "
                f"||Cu(q+d)-Cu(q)-Cu'(q)d||={norm_rem:.3e} | ||Cu(q+d)-Cu(q)||={norm_diff:.3e} | "
                f"||lin||={norm_lin:.3e} | TCC={c_tc:.3e}"
            )

            rows.append({
                "sample": k,
                "flat_index": idx,
                "row": row,
                "col": col,
                "amplitude": amplitude,
                "norm_d": norm_d,
                "norm_Cu_q": norm_Cu_q,
                "norm_Cu_qd": norm_Cu_qd,
                "norm_diff": norm_diff,
                "norm_lin": norm_lin,
                "norm_rem": norm_rem,
                "c_tc": c_tc,
            })

        c_vals = np.array([r["c_tc"] for r in rows], dtype=float)

        data = {
            "amplitude": amplitude,
            "rows": rows,
            "mean": np.nanmean(c_vals),
            "median": np.nanmedian(c_vals),
            "min": np.nanmin(c_vals),
            "max": np.nanmax(c_vals),
        }
        all_data.append(data)

        print("\nSummary")
        print("mean  :", data["mean"])
        print("median:", data["median"])
        print("min   :", data["min"])
        print("max   :", data["max"])
        print()

    create_tcc_pdf_report(all_data, pdf_filename)
    print(f"Saved plots to: {pdf_filename}")

    return all_data

def create_tcc_pdf_report(all_data, pdf_filename="tcc_analysis.pdf"):
    """
    Create a single PDF with:
      1. Summary over amplitudes
      2. Per-amplitude 2x2 pages
    """

    amplitudes = np.array([d["amplitude"] for d in all_data], dtype=float)

    means = np.array([d["mean"] for d in all_data], dtype=float)
    medians = np.array([d["median"] for d in all_data], dtype=float)
    mins = np.array([d["min"] for d in all_data], dtype=float)
    maxs = np.array([d["max"] for d in all_data], dtype=float)

    def avg_key(key):
        vals = []
        for d in all_data:
            arr = np.array([r[key] for r in d["rows"]], dtype=float)
            vals.append(np.nanmean(arr))
        return np.array(vals)

    avg_norm_d = avg_key("norm_d")
    avg_norm_rem = avg_key("norm_rem")
    avg_norm_diff = avg_key("norm_diff")
    avg_norm_lin = avg_key("norm_lin")
    avg_norm_Cu_q = avg_key("norm_Cu_q")
    avg_norm_Cu_qd = avg_key("norm_Cu_qd")

    with PdfPages(pdf_filename) as pdf:
        # ---------------------------------------------------------
        # Page 1: summary plots side by side
        # ---------------------------------------------------------
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        axs[0].loglog(amplitudes, means, "o-", label="mean")
        axs[0].loglog(amplitudes, medians, "s-", label="median")
        axs[0].loglog(amplitudes, mins, "^-", label="min")
        axs[0].loglog(amplitudes, maxs, "d-", label="max")
        axs[0].set_title(r"TCC statistics vs amplitude")
        axs[0].set_xlabel("amplitude")
        axs[0].set_ylabel(r"$c_{\mathrm{tc}}$")
        axs[0].grid(True, which="both", ls="--", alpha=0.5)
        axs[0].legend()

        axs[1].loglog(amplitudes, avg_norm_rem, "o-", label=r"$\|Cu(q+d)-Cu(q)-Cu'(q)d\|$")
        axs[1].loglog(amplitudes, avg_norm_diff, "s-", label=r"$\|Cu(q+d)-Cu(q)\|$")
        axs[1].loglog(amplitudes, avg_norm_lin, "^-", label=r"$\|Cu'(q)d\|$")
        axs[1].loglog(amplitudes, avg_norm_d, "x-", label=r"$\|d\|$")
        axs[1].set_title("Average norms vs amplitude")
        axs[1].set_xlabel("amplitude")
        axs[1].set_ylabel("norm")
        axs[1].grid(True, which="both", ls="--", alpha=0.5)
        axs[1].legend()

        fig.suptitle("TCC analysis summary", fontsize=14)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # # ---------------------------------------------------------
        # # Optional second summary page
        # # ---------------------------------------------------------
        # fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # axs[0].loglog(amplitudes, avg_norm_Cu_q, "o--", label=r"$\|Cu(q)\|$")
        # axs[0].loglog(amplitudes, avg_norm_Cu_qd, "s--", label=r"$\|Cu(q+d)\|$")
        # axs[0].set_title("Reference/output norms vs amplitude")
        # axs[0].set_xlabel("amplitude")
        # axs[0].set_ylabel("norm")
        # axs[0].grid(True, which="both", ls="--", alpha=0.5)
        # axs[0].legend()

        # ratio_lin_diff = avg_norm_lin / avg_norm_diff
        # axs[1].loglog(amplitudes, ratio_lin_diff, "o-")
        # axs[1].set_title(r"Average ratio $\|Cu'(q)d\| / \|Cu(q+d)-Cu(q)\|$")
        # axs[1].set_xlabel("amplitude")
        # axs[1].set_ylabel("ratio")
        # axs[1].grid(True, which="both", ls="--", alpha=0.5)

        # fig.suptitle("Additional summary quantities", fontsize=14)
        # fig.tight_layout()
        # pdf.savefig(fig, bbox_inches="tight")
        # plt.close(fig)

def main():
    p1 = (-0.1, -15.0, -15.0)
    p2 = ( 0.1,  15.0,  15.0)

    y_bounds = (p1[1], p2[1])
    z_bounds = (p1[2], p2[2])

    # state_y_res = 60
    # state_z_res = 60

    state_y_res = 10
    state_z_res = 10

    param_y_res = state_y_res
    param_z_res = state_z_res

    par_dim = (param_y_res + 1) * (param_z_res + 1) 
    T_initial = 0
    T_final = 16.0
    nt = 64

    delta_t = (T_final - T_initial) / nt

    rho_hat = 2.71

    assert T_final > T_initial
    q_circ = np.ones((1, par_dim))
    q_exact = np.ones((1,par_dim))
    
    half_size = 1
    q_exact = q_exact[0,:].reshape(param_y_res+1,param_z_res+1)
    add_constant_patch_coords(q_exact, 
                              center_coords=( 5.0,  0.0), 
                              value=3.0, 
                              half_size=half_size,
                              y_bounds=y_bounds, 
                              z_bounds=z_bounds)

    add_constant_patch_coords(q_exact, 
                              center_coords=(-9.0, -1.0), 
                              value=2.0, 
                              half_size=half_size,
                              y_bounds=y_bounds, 
                              z_bounds=z_bounds)

    q_exact = q_exact.flatten()
    q_exact = np.array([q_exact])

    q_circ[0,:] = 1.0
    # gamma = 0.9
    # q_circ = q_circ + gamma * (q_exact - q_circ)

    bounds = np.zeros((par_dim, 2))
    bounds[:,0] = 1e-20
    bounds[:,1] = 1e20

    state_grid_resolution = [4,state_y_res,state_z_res]
    param_grid_resolution = [4,param_y_res,param_z_res]

    setup = {
        'p1' : p1,
        'p2' : p2,
        'param_grid_resolution' : param_grid_resolution,
        'state_grid_resolution' : state_grid_resolution,
        'body_force' : {
            'type' : mm.BodyForceType.CenterExcite,
            'hyperparameter' : {
                'end_time' : 0.5,
                'factor' : (1.0 / rho_hat)
            }
        },
        'stored_energy' : {
            'type' : hm.StoredEnergyFunctionType.Hookean,
            #'type' : hm.StoredEnergyFunctionType.NeoHookean,
            'hyperparameter' : {
                # 'mu' : 26.32, 
                # 'kappa' : 68.60
                # 'mu' : 1e1, 
                # 'lambda' : 1e1
                'mu' : (5.6 / rho_hat), 
                'lambda' : (10.9 / rho_hat),
            }
        },
        'boundary_condition' : {
            'type': mm.BoundaryConditionType.DirichletOnYandZ,
            'hyperparameter' : {}
        },
        'observation_operator': {
            'type': mm.ObservationOperatorType.Identity,                       # Type of observation operator (e.g., identity = full state observed)
            #'type': mm.ObservationOperatorType.Sensors,
            'hyperparameter' : {
                # 'spatial_resolution' : state_grid_resolution,
                # 'radius' : 0.001,
                # 'second_row' : False 
            }
        },
        'dims' : {
            'nt': nt,                                     # Number of time steps
            'par_dim' : None,
            'state_dim' : None,
            'observation_space_dim': None
        },
        'products': {                                 # Inner products used in the problem
            'prod_H': 'l2',                           # Product on H_h
            'prod_Q': 'l2',                      # Product on Q_h
            #'prod_Q': 'h1',                           # Product on Q_h
            'prod_V': 'h1',                    # Product on V_h
            'prod_C': 'state_l2',                       # Product on C_h
        },
        'T_initial': T_initial,                       # Start time of the simulation
        'T_final': T_final,                           # End time of the simulation
        'delta_t': delta_t,                           # Time step size
        'noise_info' : {
            'noise_level_input' : 1.0 * 1e-2,
            #'noise_level_input' : 0.0,
            'noise_level_mode' : 'rel',
            'abs_noise_level_y' : None,
            'rel_noise_level_y' : None,
            'y_norm' : None,
        },
        'noise_level': 0,                      # Absolute noise magnitude added to data
        'q_circ': q_circ,                             # Backgroundlevel for the parameter
        'q_exact_function': None,                     # Exact parameter as function, will be set by 'build_InstationaryModelIP'
        'q_exact': q_exact,                           # Exact parameter values, will be set by 'build_InstationaryModelIP'
        'q_time_dep': False,                          # Whether parameter is time-dependent (bool)
        'riesz_rep_grad': False,                       # Use Riesz representative for gradient in optimization
        'riesz_rep_hess': False,                       
        'bounds': bounds,                             # Bounds on parameter values (e.g., for optimization)
        'save_path' : save_path,
        'time_stepper' : {
            'state' : {
                #'type' : TimeStepperType.SecondOrderCrankNicolson,
                'type' : TimeStepperType.SecondOrderCrankNicolson,
                'config' : {
                    'zeta' : 0.5
                }
            },
            'adjoint' : {
                'type' : TimeStepperType.SecondOrderCrankNicolson,
                #'type' : TimeStepperType.SecondOrderCrankNicolsonAdjointDTO,
                'config' : {
                    'zeta' : 0.5
                }
            },
            'lin_state' : {
                'type' : TimeStepperType.SecondOrderCrankNicolson,
                'config' : {
                    'zeta' : 0.5
                }
            },
            'lin_adjoint' : {
                'type' : TimeStepperType.SecondOrderCrankNicolson,
                #'type' : TimeStepperType.SecondOrderCrankNicolsonAdjointDTO,
                'config' : {
                    'zeta' : 0.5
                }
            },
        }
    }


    FOM = build_HyperElasticityModelIP(setup, logger)
    q_exact = FOM.setup['q_exact']
    q_start = q_circ


    # nabla_J = FOM.compute_gradient(FOM.Q.make_array(q_exact))
    # print(np.sqrt(FOM.products['prod_Q'].apply2(nabla_J, nabla_J)))

    # import sys
    # sys.exit()


    # u_exact = FOM.solve_state(FOM.Q.make_array(q_exact))
    # FOM.A.hyperelasticity_model.save_time_series(
    #     [v.impl for v in u_exact.vectors],
    #     str('u_exact'),
    #     str(save_path),
    #     np.linspace(T_initial, T_final, nt+1)
    # )

    # p_exact = FOM.solve_adjoint(FOM.Q.make_array(q_exact), u = u_exact)
    # FOM.A.hyperelasticity_model.save_time_series(
    #     [v.impl for v in p_exact.vectors],
    #     str('p_exact'),
    #     str(save_path),
    #     np.linspace(T_initial, T_final, nt+1)
    # )

    # u_start = FOM.solve_state(FOM.Q.make_array(q_start))
    # FOM.A.hyperelasticity_model.save_time_series(
    #     [v.impl for v in u_start.vectors],
    #     str('u_start'),
    #     str(save_path),
    #     np.linspace(T_initial, T_final, nt+1)
    # )


    # p_start = FOM.solve_adjoint(FOM.Q.make_array(q_start), u = u_start)
    # FOM.A.hyperelasticity_model.save_time_series(
    #     [v.impl for v in p_start.vectors],
    #     str('p_start'),
    #     str(save_path),
    #     np.linspace(T_initial, T_final, nt+1)
    # )

    # diff = u_start - u_exact
    # FOM.A.hyperelasticity_model.save_time_series(
    #     [v.impl for v in diff.vectors],
    #     str('diff'),
    #     str(save_path),
    #     np.linspace(T_initial, T_final, nt+1)
    # )

    model = NormDiffModel(
        q_circ = FOM.q_circ,
        q_exact = FOM.Q.make_array(FOM.setup['q_exact']),
        Q = FOM.Q,
        C = FOM.C,
        products = FOM.products,
        setup = FOM.setup,
        nt = FOM.nt
    )
    
    all_data = run_tcc_analysis(
        FOM,
        q=FOM.q_circ,
        amplitudes=[1e0, 1e-2, 1e-4, 1e-6, 1e-8,1e-10,1e-12],        
        max_h=20,
        seed=0,
        use_gradient_direction=True,   # set False for node-wise localized perturbations
        pdf_filename="tcc_analysis.pdf",
    )


if __name__ == '__main__':
    main()
