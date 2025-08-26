import numpy as np
import logging
import os
from pathlib import Path
from datetime import datetime

from pymor.basic import *

import RBInvParam.problems.elasticity.material_model as mm

from RBInvParam.optimizer import FOMOptimizer
from RBInvParam.utils.io import save_dict_to_pkl
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.problems.elasticity.build import build_InstationaryModelIP


#########################################################################################''

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = Path('./dumps') / (timestamp + '_FOM_IRGNM')
os.mkdir(save_path)
logfile_path= save_path / 'FOM_IRGNM.log'

logger = get_default_logger(logger_name='FOM_IRGNM',
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

    T_initial = 0
    T_final = 4
    nt = 16
    delta_t = (T_final - T_initial) / nt

    assert T_final > T_initial
    q_circ = np.ones((1, par_dim))
    q_exact = np.ones((1,par_dim))
    q_exact[0,27] = 2
    q_exact[0,54] = 3

    bounds = np.zeros((par_dim, 2))
    bounds[:,0] = 0.001
    bounds[:,1] = 1e20
    # bounds[:,0] = -1e10
    # bounds[:,1] = 1e10


    setup = {
        'spatial_resolution' : [4,y_res,z_res],
        'body_force_type' : mm.BodyForceType.CenterExcite,
        'system_matrix' : {
            'type' : mm.SystemMatrixType.CosseratDelamination,
            'hyperparameter' : {
                'lambda' : 12.0,
                'mu' : 8.0,
                'nu' : 1.0,
                'surface' : 'left'
            }
        },
        'observation_operator': {
            'type': mm.ObservationOperatorType.SensorsR8d,                       # Type of observation operator (e.g., identity = full state observed)
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
        'noise_level': 0.0,                             # Absolute noise magnitude added to data
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

    print(FOM.compute_objective(FOM.Q.make_array(q_start)))
    print(FOM.compute_objective(FOM.Q.make_array(q_exact)))

    optimizer_parameter = {
        'q_0': q_start,                                          # Initial guess for the parameter to be optimized
        'alpha_0': 1e-5,                                          # Initial regularization parameter
        'tol': 1e-11,                                            # Absolute convergence tolerance for optimization
        'tau': 3.5,                                              # Relative (to the noise) convergence tolerance for optimization
        'noise_level': setup['noise_level'],                     # Noise level in observed data (from model setup)
        'theta': 0.1,                                           # Lower tolerance for the direction acceptance condition
        'Theta': 1.95,                                           # Upper tolerance for the direction acceptance condition
        #####################
        'i_max': 250,                                             # Maximum number of outer optimization iterations
        'reg_loop_max': 25,                                      # Maximum number of regularization updates per step
        'i_max_inner': 10,                                       # Maximum number of inner iterations
        ####################
        'lin_solver_parms': {
            'method' : 'gd',                                     # Method for solving linear systems (e.g., gradient descent)
            'max_iter': 1e4,                                     # Max iterations for the linear solver
            'lin_solver_tol': 1e-9,                          # Tolerance for convergence in the linear solver
            'inital_step_size': 1                                # Initial step size for iterative solvers (if applicable)
        },
        'use_cached_operators': True ,                          # Whether to reuse assembled operators (improves speed if True)
        'dump_every_nth_loop': 2,                                # Dump intermediate results every n optimization iterations
    }


    logger.info(f"Dumping model setup to {save_path / 'setup.pkl'}.")
    save_dict_to_pkl(path=save_path / 'setup.pkl', 
                     data = setup,
                     use_timestamp=False)
        
    logger.info(f"Dumping model optimizer_parameter to {save_path / 'optimizer_parameter.pkl'}.")
    save_dict_to_pkl(path=save_path / 'optimizer_parameter.pkl', 
                        data = optimizer_parameter,
                        use_timestamp=False)

    optimizer = FOMOptimizer(
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
