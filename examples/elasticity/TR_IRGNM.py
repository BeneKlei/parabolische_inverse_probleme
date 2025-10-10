import numpy as np
import logging
import os
from pathlib import Path
from datetime import datetime

from pymor.basic import *

import RBInvParam.problems.elasticity.material_model as mm

from RBInvParam.optimizer import QrVrROMOptimizer
from RBInvParam.utils.io import save_dict_to_pkl
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.problems.elasticity.build import build_InstationaryModelIP

from RBInvParam.error_estimators.state_error_estimators import StateErrorEstimatorType
from RBInvParam.error_estimators.adjoint_error_estimators import AdjointErrorEstimatorType
from RBInvParam.error_estimators.objective_error_estimators import ObjectiveErrorEstimatorType

#########################################################################################

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
    'pymor' : 'WARN'
})

set_defaults({})

#########################################################################################''

def main():
    y_res = 30
    z_res = 30

    par_dim = (y_res + 1) * (z_res + 1) 
    #* 5 * 3
    #par_dim = 3
    T_initial = 0
    #T_final = 10.0
    # T_final = 5.0
    # nt = 50

    # T_final = 5.0
    # nt = 50

    T_final = 5.0
    nt = 50

    # T_final = 10.0
    # nt = 100

    delta_t = (T_final - T_initial) / nt

    assert T_final > T_initial
    q_circ = np.ones((1, par_dim)) * 1
    q_exact = np.ones((1,par_dim)) * 1

    #q_exact[0,40] = 4e2

    q_exact[0,200] = 2
    q_exact[0,300] = 3

    #q_exact[0,50] = 2
    q_circ[0,:] = 1

    bounds = np.zeros((par_dim, 2))
    bounds[:,0] = 1e-20
    bounds[:,1] = 1e20


    setup = {
        'spatial_resolution' : [4,y_res,z_res],
        'body_force_type' : mm.BodyForceType.CenterExcite,
        'system_matrix' : {
            'type' : mm.SystemMatrixType.CosseratDelamination,
            'hyperparameter' : {
                # 'lambda' : 1e2,
                # 'mu' : 1e2,
                'lambda' : 1e1,
                'mu' : 1e1,
                'nu' : 1e-3,
                'surface' : 'left'
            }
        },
        'observation_operator': {
            'type': mm.ObservationOperatorType.SensorsR28d,                       # Type of observation operator (e.g., identity = full state observed)
            #'type': mm.ObservationOperatorType.SensorsR56d,                       # Type of observation operator (e.g., identity = full state observed)
            #'type': mm.ObservationOperatorType.Identity,                       # Type of observation operator (e.g., identity = full state observed)
            #'type': mm.ObservationOperatorType.Boundary,                       # Type of observation operator (e.g., identity = full state observed)
            'hyperparameter' : {}
        },
        'dims' : {
            'nt': nt,                                     # Number of time steps
            'par_dim' : None,
            'state_dim' : None,
            'observation_space_dim': None
        },
        'products': {                                 # Inner products used in the problem
            'prod_H': 'l2',                           # Product on H_h
            'prod_Q': 'euclid',                       # Product on Q_h
            'prod_V': 'h1_0_semi',                    # Product on V_h
            #'prod_V': 'l2',
            'prod_C': 'euclid',                       # Product on C_h
        },
        'T_initial': T_initial,                       # Start time of the simulation
        'T_final': T_final,                           # End time of the simulation
        'delta_t': delta_t,                           # Time step size
        'noise_percentage': None,                     # Relative noise level, will be set by 'build_InstationaryModelIP'
        'noise_level': 5 * 1e-5,                      # Absolute noise magnitude added to data
        #'noise_level': 0.0,                      # Absolute noise magnitude added to data
        'q_circ': q_circ,                             # Backgroundlevel for the parameter
        'q_exact_function': None,                     # Exact parameter as function, will be set by 'build_InstationaryModelIP'
        'q_exact': q_exact,                           # Exact parameter values, will be set by 'build_InstationaryModelIP'
        'q_time_dep': False,                          # Whether parameter is time-dependent (bool)
        'riesz_rep_grad': True,                       # Use Riesz representative for gradient in optimization
        'bounds': bounds,                             # Bounds on parameter values (e.g., for optimization)
        'save_path' : save_path,
        'time_stepper' : {
            'name' : 'newman_second_order',
            'zeta' : 0.5
        }
    }

    FOM = build_InstationaryModelIP(setup, logger)
    q_exact = FOM.setup['q_exact']
    q_start = q_circ

    u_exact = FOM.solve_state(FOM.Q.make_array(q_exact))
    FOM.A.material_model.save_time_series(
        [v.real_part.impl for v in u_exact.vectors],
        str('u_exact'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )

    p_exact = FOM.solve_adjoint(FOM.Q.make_array(q_exact), u = u_exact)
    FOM.A.material_model.save_time_series(
        [v.real_part.impl for v in p_exact.vectors],
        str('p_exact'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )

    u_start = FOM.solve_state(FOM.Q.make_array(q_start))
    FOM.A.material_model.save_time_series(
        [v.real_part.impl for v in u_start.vectors],
        str('u_start'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )

    p_start = FOM.solve_adjoint(FOM.Q.make_array(q_start), u = u_start)
    FOM.A.material_model.save_time_series(
        [v.real_part.impl for v in p_start.vectors],
        str('p_start'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )

    diff = u_start - u_exact
    FOM.A.material_model.save_time_series(
        [v.real_part.impl for v in diff.vectors],
        str('diff'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )


    optimizer_parameter = {
        'q_0': q_start,                                              # Initial guess for the parameter to be optimized        
        'alpha_0': 1e-5,                                              # Initial regularization parameter (data fidelity vs. regularization)        
        #'alpha_0': 1e-14,                                              # Initial regularization parameter (data fidelity vs. regularization)        
        'tol': 1e-9,                                                 # Absolute convergence tolerance for optimization
        #'tau': 1.50,                                                  # Relative (to the noise) convergence tolerance for optimization
        'tau': 1.00,                                                  # Relative (to the noise) convergence tolerance for optimization
        'noise_level': setup['noise_level'],                         # Noise level in observed data (from model setup)
        'theta': 0.4,
        'Theta': 1.95,                                               # Upper bound for step acceptance condition
        'tau_tilde': 3.5,                                            # Relative (to the noise) convergence tolerance for optimization inside the trust region
        #####################
        'i_max': 250,                                                 # Max number of outer optimization iterations
        'reg_loop_max': 10,                                          # Max number of regularization updates per iteration
        'i_max_inner': 30,                                           # Max number of inner iterations
        'agc_armijo_max_iter': 100,                                  # Max iterations for computing the AGC
        'TR_armijo_max_iter': 10,                                     # Max iterations Armijo condition to enforce the trust-region 
        #####################
        'lin_solver_parms': {
            'method': 'gd',                                          # Method for solving linear systems (e.g., gradient descent)
            'max_iter': 1e3,                                         # Maximum iterations for the linear solver
            #'lin_solver_tol': 1e-8,                                 # Convergence tolerance for the linear solver
            'lin_solver_tol': 1e-12,                                 # Convergence tolerance for the linear solver
            'inital_step_size': 1                                    # Initial step size for iterative linear solver
        },
        # 'lin_solver_parms': {
        #     'method': 'BiCGSTAB',                                  # BiCGSTAB method for solving nonsymmetric linear systems
        #     'rtol': 1e-12,                                         # Relative convergence tolerance
        #     'atol': 1e-12,                                         # Absolute convergence tolerance
        #     'maxiter': 1e3                                         # Max iterations for BiCGSTAB solver
        # },
        'enrichment': {
            'parameter_basis' : {
                'strategy': 'snapshot_HaPOD',                  
                'HaPOD_tol': 1e-16,
                'transformation' : {
                    'sample_every_n_th' : 1,
                    'normalize': False,
                }
            },
            'state_basis' : {
                'strategy': 'snapshot_HaPOD',                      # Enrichment strategy for state basis
                'HaPOD_tol': 1e-6,
                'transformation' : {
                    'sample_every_n_th' : 1,
                    'normalize': True
                }
            }
        },
        'error_estimator_types' : {
            'state' : StateErrorEstimatorType.NONE,
            'adjoint' : AdjointErrorEstimatorType.NONE,
            'objective' : ObjectiveErrorEstimatorType.NONE,
        },
        #####################
        'use_cached_operators': True,                               # Reuse previously assembled operators to save computation
        'dump_every_nth_loop': 1,                                    # Dump intermediate results every n optimization iterations
        #####################
        #'eta0': 1e-1,                                                # Initial trust region tolerance
        'eta0': 1e-2,                                                # Initial trust region tolerance
        'kappa_arm': 1e-12,                                          # Armijo condition constant for sufficient decrease
        'beta_1': 0.90,                                              # Trust region edge tolerance.
        'beta_2': 3/4,                                               # Tolerance for the trustworthiness. 
        'beta_3': 0.5                                                # Shrinking/Enlarging factor for the trust region.
    }

    logger.info(f"Dumping model setup to {save_path / 'setup.pkl'}.")
    save_dict_to_pkl(path=save_path / 'setup.pkl', 
                     data = setup,
                     use_timestamp=False)
        
    logger.info(f"Dumping model optimizer_parameter to {save_path / 'optimizer_parameter.pkl'}.")
    save_dict_to_pkl(path=save_path / 'optimizer_parameter.pkl', 
                        data = optimizer_parameter,
                        use_timestamp=False)

    optimizer = QrVrROMOptimizer(
        FOM = FOM,
        optimizer_parameter = optimizer_parameter,
        logger = logger,
        save_path=save_path
    )
    q_est = optimizer.solve()
    print(q_est)
    u = FOM.solve_state(q_est)
    FOM.A.material_model.save_time_series(
        [v.real_part.impl for v in u.vectors],
        str('u_est'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )

    diff = u - u_exact
    FOM.A.material_model.save_time_series(
        [v.real_part.impl for v in diff.vectors],
        str('diff_est'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )



if __name__ == '__main__':
    main()
