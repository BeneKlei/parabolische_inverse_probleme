import numpy as np
import logging
import argparse
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("./dumps"),
        help="Root directory where timestamped dump folders will be created."
    )
    return parser.parse_args()

args = parse_args()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = args.working_dir / f"{timestamp}_batch_TR_IRGNM"
save_path.mkdir(parents=True, exist_ok=False)

# optional root logger for batch-level messages
root_logfile_path = save_path / "batch.log"
logger = get_default_logger(
    logger_name="batch_TR_IRGNM",
    logfile_path=root_logfile_path,
    use_timestemp=False
)
logger.setLevel(logging.DEBUG)

#########################################################################################''
set_log_levels({
    'pymor.operators.constructions.LincombOperator' : 'ERROR',
    #'pymor.operators.constructions.AdjointOperator' : 'ERROR',
    #'pymor.algorithms.genericsolvers.lgmres' : 'ERROR',
    'pymor.algorithms' : 'ERROR'
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

def make_optimization_logger(base_name: str, log_dir: Path, idx: int):
    logfile_path = log_dir / f"optimization_{idx:03d}.log"
    logger = get_default_logger(
        logger_name=f"{base_name}_{idx:03d}",
        logfile_path=logfile_path,
        use_timestemp=False
    )
    logger.setLevel(logging.DEBUG)
    return logger

def solve_problem_batch(
    setup : Dict,
    optimizer_parameter : Dict,
    q_exacts: List[np.ndarray],
    q_circ_inital : np.ndarray,
    batch_save_path: Path,
    batch_logger,
    reset_q_circ: bool = True
) -> List[np.ndarray]:

    q_circ = q_circ_inital.copy()
    q_ests = []

    batch_logger.info(f"Starting batch solve with {len(q_exacts)} optimization runs.")
    batch_logger.info(f"Batch root directory: {batch_save_path}")

    
    for i, q_exact in enumerate(q_exacts):
        opt_dir = batch_save_path / f"opt_{i:03d}"
        opt_dir.mkdir(parents=True, exist_ok=True)

        opt_logger = make_optimization_logger("TR_IRGNM_opt", opt_dir, i)
        opt_logger.info("=" * 80)
        opt_logger.info(f"Starting optimization {i + 1}/{len(q_exacts)}")
        opt_logger.info(f"Optimization directory: {opt_dir}")

        local_setup = copy.deepcopy(setup)
        local_optimizer_parameter = copy.deepcopy(optimizer_parameter)

        local_setup["q_circ"] = q_circ.copy()
        local_setup["q_exact"] = q_exact
        local_setup["save_path"] = opt_dir

        # print(local_setup["q_exact"])

        # import matplotlib.pyplot as plt
        # plt.imshow(local_setup["q_exact"].reshape(31,31))
        # plt.show()

        # import sys
        # sys.exit()

        FOM = build_HyperElasticityModelIP(local_setup, opt_logger)
    
        save_dict_to_pkl(
            path=opt_dir / "setup.pkl",
            data=local_setup,
            use_timestamp=False
        )

        local_optimizer_parameter["q_0"] = q_circ.copy()
        local_optimizer_parameter["noise_level"] = local_setup["noise_info"]["abs_noise_level_y"]

        save_dict_to_pkl(
            path=opt_dir / "optimizer_parameter.pkl",
            data=local_optimizer_parameter,
            use_timestamp=False
        )

        optimizer = QrVrROMOptimizer(
            FOM=FOM,
            optimizer_parameter=local_optimizer_parameter,
            logger=opt_logger,
            save_path=opt_dir
        )

        if len(q_ests) > 0:
            opt_logger.info("Adding previous estimate as initial parameter-basis snapshot.")
            optimizer.add_initial_snapshots(
                basis="parameter_basis",
                snapshots=FOM.Q.make_array(q_ests[-1])
            )

        q_est = optimizer.solve()
        q_est = q_est.to_numpy()
        q_ests.append(q_est)

        opt_logger.info(f"Finished optimization {i + 1}/{len(q_exacts)}")

        if not reset_q_circ:
            print(type(q_est))
            print(q_est)
            q_circ = q_est.copy()
            opt_logger.info("Updated q_circ for next run from current q_est.")
        else:
            opt_logger.info("Resetting q_circ behavior enabled; keeping original initial q_circ.")

        batch_logger.info(f"Finished optimization {i + 1}/{len(q_exacts)} in {opt_dir}")

    batch_logger.info("Finished all batch optimizations.")
    return q_ests
    
def main():
    p1 = (-0.1, -15.0, -15.0)
    p2 = ( 0.1,  15.0,  15.0)

    center = (
        p1[0],
        p1[1] + (p2[1] - p1[1]) / 2,
        p1[2] + (p2[2] - p1[2]) / 2        
    )

    y_bounds = (p1[1], p2[1])
    z_bounds = (p1[2], p2[2])

    state_y_res = 30
    state_z_res = 30

    # state_y_res = 60
    # state_z_res = 60

    param_y_res = state_y_res
    param_z_res = state_z_res

    par_dim = (param_y_res + 1) * (param_z_res + 1) 

    #################################################
    # Set:
    # 1 PU = 10^9 GPa
    # 1 LU = 1/30m
    # 1 DU = 10^3 kg m^{-3}
    # Derived
    # 1 TU = 3.33 * 10^-5s
    #################################################

    T_initial = 0    
    T_final = 16.0
    nt = 32

    delta_t = (T_final - T_initial) / nt
    
    rho_hat = 2.70
    parameter_factor = 10

    assert T_final > T_initial
    q_circ = parameter_factor  * np.ones((1, par_dim))

    bounds = np.zeros((par_dim, 2))
    bounds[:,0] = 1e-20
    bounds[:,1] = 1e20

    state_grid_resolution = [4,state_y_res,state_z_res]
    param_grid_resolution = [4,param_y_res,param_z_res]

    q_exacts = add_constant_rect_patch_from_corners_coords_variations(
        tl_coords = (-10, -10),
        br_coords = (-10 + 5, -10 + 20),
        value = 0.5,
        param_y_res = param_y_res,
        param_z_res = param_z_res,
        parameter_factor = parameter_factor
    )

    setup = {
        'p1' : p1,
        'p2' : p2,
        'param_grid_resolution' : param_grid_resolution,
        'state_grid_resolution' : state_grid_resolution,
        'body_force' : {
            'type' : mm.BodyForceType.SharpPulse,
            'hyperparameter' : {
                'origin' : center,
                'end_time' : 0.5,
                'factor' : (1.0 / rho_hat),
                'width' : 1.00
            }
        },
        'stored_energy' : {
            'type' : hm.StoredEnergyFunctionType.Hookean,
            #'type' : hm.StoredEnergyFunctionType.NeoHookean,
            'hyperparameter' : {
                # 'mu' : 26.32, 
                # 'kappa' : 68.60
                # 'mu' : (5.6 / rho_hat), 
                # 'lambda' : (10.9 / rho_hat),
                # 'mu' : (5.6 / rho_hat), 
                # 'lambda' : (10.9 / rho_hat),
                'mu' : 1/parameter_factor * (11.2 / rho_hat), 
                'lambda' : 1/parameter_factor *  (21.8 / rho_hat),
                # 'mu' : 4 * 4.15,
                # 'lambda' : 4 * 8.07
            }
        },
        'boundary_condition' : {
            'type': mm.BoundaryConditionType.DirichletOnYandZ,
            'hyperparameter' : {}
        },
        'observation_operator': {
            'type': mm.ObservationOperatorType.Identity,                       # Type of observation operator (e.g., identity = full state observed)
            'hyperparameter' : {},
            # 'type': mm.ObservationOperatorType.Sensors,
            # 'hyperparameter' : {
            #     'p1' : p1,
            #     'p2' : p2,
            #     'sensor_patch_size' : (28.0, 28.0),
            #     'sensor_spacing' : 1.0,
            #     'at_top' : True,
            #     'at_bottom' : False,
            #     'sensor_patch_center_offset' : (0.0, 0.0),
            #     'x_face_offset' : 0.00,
            #     'radius' : 0.001,  
            #     'use_boundary_mass_matrix' : True,
            # }
        },
        'dims' : {
            'nt': nt,                                     # Number of time steps
            'par_dim' : None,
            'state_dim' : None,
            'observation_space_dim': None
        },
        'products': {                                 # Inner products used in the problem
            'prod_H': 'l2',                           # Product on H_h
            'prod_Q': 'euclid',                      # Product on Q_h
            'prod_V': 'h1',                           # Product on V_h            
            'prod_C': 'euclid',                       # Product on C_h
        },
        'T_initial': T_initial,                       # Start time of the simulation
        'T_final': T_final,                           # End time of the simulation
        'delta_t': delta_t,                           # Time step size
        'noise_percentage': None,                     # Relative noise level, will be set by 'build_InstationaryModelIP'
        'noise_info' : {
            'noise_level_input' : 1.0 * 1e-2,
            'noise_level_mode' : 'rel',
            'abs_noise_level_y' : None,
            'rel_noise_level_y' : None,
            'y_norm' : None,
        },
        'q_circ': None,                             # Backgroundlevel for the parameter
        'q_exact_function': None,                     # Exact parameter as function, will be set by 'build_InstationaryModelIP'
        'q_exact': None,                           # Exact parameter values, will be set by 'build_InstationaryModelIP'
        'q_time_dep': False,                          # Whether parameter is time-dependent (bool)
        'riesz_rep_grad': True,                       # Use Riesz representative for gradient in optimization
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

    optimizer_parameter = {
        'method' : 'TR_IRGNM',
        'q_0': None,                                              # Initial guess for the parameter to be optimized
        'alpha_0': 1e-5,                                              # Initial regularization parameter (data fidelity vs. regularization)
        #'alpha_0': 1e-10,                                              # Initial regularization parameter (data fidelity vs. regularization)
        'tol': 1e-9,                                                 # Absolute convergence tolerance for optimization
        #'tau': 1.25,                                                  # Relative (to the noise) convergence tolerance for optimization
        'tau': 2.0,                                                  # Relative (to the noise) convergence tolerance for optimization
        #'noise_level': setup['noise_info']['abs_noise_level_y'],                         # Noise level in observed data (from model setup)
        'noise_level': None,
        'theta': 0.40,
        'Theta': 1.95,                                               # Upper bound for step acceptance condition
        #'Theta': 1.50,                                               # Upper bound for step acceptance condition
        'tau_tilde': 3.5,                                            # Relative (to the noise) convergence tolerance for optimization inside the trust region
        #####################
        'i_max': 250,                                                 # Max number of outer optimization iterations
        'reg_loop_max': 15,                                          # Max number of regularization updates per iteration
        #'i_max_inner': 15,                                           # Max number of inner iterations
        'i_max_inner': 30,                                           # Max number of inner iterations
        'AGC_armijo_cfg' : {
            "max_iter": 50,
            "initial_step_size": 1.0,
            "kappa_arm": 1e-12,
            "shrink": 0.5,
        },
        'TR_armijo_cfg' : {
            "max_iter": 5,
            "initial_step_size": 1.0,
            "kappa_arm": 1e-12,
            "shrink": 0.5,
        },
        'TR': {
            'type': TRType.RADIUS,

            # TR config
            'eta_initial': parameter_factor * 1e10,
            'eta_min': parameter_factor * 1e-5,
            'eta_max': parameter_factor * 1e10,
            'beta_1': 0.80,
            'beta_2': 0.80,
            'beta_3': 0.75,
        },
        # 'TR': {
        #     'type': TRType.RADIUS,

        #     # TR config
        #     'eta_initial': 0.25,        
        #     'eta_min': 1e-2,
        #     'eta_max': 1.00,
        #     'beta_1': 0.80,
        #     'beta_2': 0.80,
        #     'beta_3': 0.75,
        # },
        #####################
        'use_cached_operators': True,                               # Reuse previously assembled operators to save computation
        'use_error_estimator' : False,
        'reg_AGC_step' : False,
        'TR_enforcement' : 'backtracking',
        'dump_every_nth_loop': 1,                                    # Dump intermediate results every n optimization iterations
        'reductor' : {
            #'type' : 'default',
            'type' : 'material_model',
            'use_adjoint_space' : False,
            'offline_parallel' : False,
            'error_estimator_types' : {
                'state' : StateErrorEstimatorType.HYPERBOLIC,
                'adjoint' : AdjointErrorEstimatorType.NONE,
                'objective' : ObjectiveErrorEstimatorType.NAIVE,
            },
            'check_orthonormality' : True,
            'check_tol' : 1e-9,
            'linearization_method' : LinearizationMethod.DEIM,
        },
        #####################
        'lin_solver_parms': {
            'method': 'gd',                                          # Method for solving linear systems (e.g., gradient descent)
            'max_iter': 250,                                         # Maximum iterations for the linear solver
            #'abs_grad_tol' : 5 * 1e-9,
            'abs_grad_tol' : 5 * 1e-11,
            'rel_change_obj_tol' : 1e-4,
            'kappa_arm' : 1e-12,
            'armijo_inital_step_size': 1e-2,                                    # Initial step size for iterative linear solver
            'armijo_min_step_size' : 1e-20
        },
        'enrichment': {
            'parameter_basis' : {
                'additional_snapshots' :{
                    'include_lin_grad' : False,
                    'include_each_nabla_J_time_step' : False,
                    'include_each_nabla_lin_J_time_step' : False,
                    'include_krylov_directions' : False,
                    'include_q_exact' : False
                },
                'compression' :
                {
                    'normalize' : True,
                    'HaPOD' : 
                    {
                        'eps': 1e-1,
                        'omega' : 0.1,
                    },
                    'every_n' : None,
                },
                'coarsing' : None,
            },
            'state_basis' : None,
            # {
            #     'additional_snapshots' :{
            #         'include_lin_states' : False,
            #         'include_krylov_sensitivites' : False,
            #     },
            #     'compression' : 
            #     {
            #         'normalize' : True,
            #         'HaPOD' : {
            #             'eps': 1e-3,
            #             'omega' : 0.1,
            #         },
            #         'every_n' : None,
            #         # 'normalize' : None,
            #         # 'HaPOD' : None,
            #     },
            #     'coarsing' : None,
            #     # 'coarsing' : {
            #     #     'rel_tol_coeff_u' : 1e-2,
            #     #     'rel_tol_coeff_p' : 1e-2
            #     # }
            # },
            'adjoint_basis' : None
        },
        'logging' : {
            'errors' : LoggerErrorChoice.OBJECTIVE,
        },        
    }

    q_ests = solve_problem_batch(
        setup=setup,
        optimizer_parameter=optimizer_parameter,
        q_exacts=q_exacts,
        q_circ_inital=q_circ,
        batch_save_path=save_path,
        batch_logger=logger,
        reset_q_circ=True
    )

if __name__ == '__main__':
    main()
