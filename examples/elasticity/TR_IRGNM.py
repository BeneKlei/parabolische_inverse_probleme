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
    y_res = 8
    z_res = 8
    par_dim = (y_res + 1) * (z_res + 1) * 5 * 3
    T_initial = 0
    T_final = 1
    nt = 50
    delta_t = (T_final - T_initial) / nt

    assert T_final > T_initial
    q_circ = np.ones((1, par_dim))
    q_exact = np.ones((1,par_dim))
    # q_exact[0,27] = 2
    # q_exact[0,54] = 3
    q_circ[0,:] = 3

    bounds = np.zeros((par_dim, 2))
    bounds[:,0] = 0.001
    bounds[:,1] = 1e20
    # bounds[:,0] = -1e10
    # bounds[:,1] = 1e10


    setup = {
        'spatial_resolution' : [4,y_res,z_res],
        'body_force_type' : mm.BodyForceType.CenterExcite,
        'system_matrix' : {
            'type' : mm.SystemMatrixType.CosseratSpatial,
            'hyperparameter' : {
                'lambda' : 12.0,
                'mu' : 8.0,
                'nu' : 1.0,
                'surface' : 'left'
            }
        },
        'observation_operator': {
            'type': mm.ObservationOperatorType.SensorsR9d,                       # Type of observation operator (e.g., identity = full state observed)
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
            'prod_C': 'euclid',                       # Product on C_h
        },
        'T_initial': T_initial,                       # Start time of the simulation
        'T_final': T_final,                           # End time of the simulation
        'delta_t': delta_t,                           # Time step size
        'noise_percentage': None,                     # Relative noise level, will be set by 'build_InstationaryModelIP'
        'noise_level': 1e-5,                           # Absolute noise magnitude added to data
        'q_circ': q_circ,                             # Backgroundlevel for the parameter
        'q_exact_function': None,                     # Exact parameter as function, will be set by 'build_InstationaryModelIP'
        'q_exact': q_exact,                           # Exact parameter values, will be set by 'build_InstationaryModelIP'
        'q_time_dep': False,                          # Whether parameter is time-dependent (bool)
        'riesz_rep_grad': True,                       # Use Riesz representative for gradient in optimization
        'bounds': bounds,                             # Bounds on parameter values (e.g., for optimization)
        'time_stepper' : {
            'name' : 'newman_second_order',
            'zeta' : 0.5
        }
    }

    FOM = build_InstationaryModelIP(setup, logger)
    q_exact = FOM.setup['q_exact']
    q_start = q_circ


    optimizer_parameter = {
        'q_0': q_start,                                              # Initial guess for the parameter to be optimized
        'alpha_0': 1e-7,                                             # Initial regularization parameter (data fidelity vs. regularization)
        'tol': 1e-9,                                                 # Absolute convergence tolerance for optimization
        'tau': 3.5,                                                  # Relative (to the noise) convergence tolerance for optimization
        'noise_level': setup['noise_level'],                         # Noise level in observed data (from model setup)
        'theta': 0.4,                                                # Lower bound for step acceptance condition
        'Theta': 1.95,                                               # Upper bound for step acceptance condition
        'tau_tilde': 3.5,                                            # Relative (to the noise) convergence tolerance for optimization inside the trust region
        #####################
        'i_max': 75,                                                 # Max number of outer optimization iterations
        'reg_loop_max': 10,                                          # Max number of regularization updates per iteration
        'i_max_inner': 2,                                           # Max number of inner iterations
        'agc_armijo_max_iter': 100,                                  # Max iterations for computing the AGC
        'TR_armijo_max_iter': 5,                                     # Max iterations Armijo condition to enforce the trust-region 
        #####################
        'lin_solver_parms': {
            'method': 'gd',                                          # Method for solving linear systems (e.g., gradient descent)
            'max_iter': 1e4,                                         # Maximum iterations for the linear solver
            'lin_solver_tol': 1e-10,                                 # Convergence tolerance for the linear solver
            'inital_step_size': 1                                    # Initial step size for iterative linear solver
        },
        # 'lin_solver_parms': {
        #     'method': 'BiCGSTAB',                                  # BiCGSTAB method for solving nonsymmetric linear systems
        #     'rtol': 1e-12,                                         # Relative convergence tolerance
        #     'atol': 1e-12,                                         # Absolute convergence tolerance
        #     'maxiter': 1e3                                         # Max iterations for BiCGSTAB solver
        # },
        'enrichment': {
            'parameter_strategy': 'snapshot_HaPOD',                  # Enrichment strategy for parameter basis
            'parameter_HaPOD_tol': 1e-9,                             # Tolerance for parameter basis POD
            'state_strategy': 'snapshot_HaPOD',                      # Enrichment strategy for state basis
            'state_HaPOD_tol': 1e-9                                  # Tolerance for state basis POD
        },
        #####################
        'use_cached_operators': True,                               # Reuse previously assembled operators to save computation
        'dump_every_nth_loop': 2,                                    # Dump intermediate results every n optimization iterations
        #####################
        'eta0': 1e-1,                                                # Initial trust region tolerance
        'kappa_arm': 1e-12,                                          # Armijo condition constant for sufficient decrease
        'beta_1': 0.95,                                              # Trust region edge tolerance.
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
    # logger.debug("Differnce to q_exact:")
    # logger.debug("L^inf") 
    # delta_q = q_est - q_exact
    # logger.debug(f"  {np.max(np.abs(delta_q.to_numpy())):3.4e}")
    
    # if q_time_dep:
    #     norm_delta_q = np.sqrt(FOM.products['bochner_prod_Q'].apply2(delta_q, delta_q))[0,0]
    #     norm_q_exact = np.sqrt(FOM.products['bochner_prod_Q'].apply2(q_exact, q_exact))[0,0]
    # else:
    #     norm_delta_q = np.sqrt(FOM.products['prod_Q'].apply2(delta_q, delta_q))[0,0]
    #     norm_q_exact = np.sqrt(FOM.products['prod_Q'].apply2(q_exact, q_exact))[0,0]
    
    # logger.debug(f"  Absolute error: {norm_delta_q:3.4e}")
    # logger.debug(f"  Relative error: {norm_delta_q / norm_q_exact * 100:3.4}%.")

if __name__ == '__main__':
    main()
