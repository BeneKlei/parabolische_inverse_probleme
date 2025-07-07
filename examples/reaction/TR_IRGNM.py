import numpy as np
import logging
import os
from pathlib import Path
from datetime import datetime

from pymor.basic import *

from RBInvParam.optimizer import QrVrROMOptimizer
from RBInvParam.utils.io import save_dict_to_pkl
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.problems.problems import build_InstationaryModelIP

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
    #N = 300
    #N = 100
    N = 30
    par_dim = (N+1)**2
    fine_N = 2 * N


    T_initial = 0
    T_final = 1
    # TODO Here is a Bug
    nt = 50
    #nt = 100
    delta_t = (T_final - T_initial) / nt
    #q_time_dep = False
    q_time_dep = True

    noise_level = 1e-5

    assert T_final > T_initial
    if q_time_dep:
        q_circ = 3*np.ones((nt, par_dim))
        bounds = np.zeros((nt * par_dim, 2))
    else:
        q_circ = 3*np.ones((1, par_dim))
        bounds = np.zeros((par_dim, 2))

    bounds[:,0] = 0.001
    bounds[:,1] = 1e3

    setup = {
        'dims': {
            'N': N,                                       # Coarse spatial discretization parameter (grid resolution)
            'nt': nt,                                     # Number of time steps
            'fine_N': fine_N,                             # Fine spatial discretization parameter (higher resolution than N)
            'state_dim': (N+1)**2,                        # Total number of spatial degrees of freedom for coarse grid
            'fine_state_dim': (fine_N+1)**2,              # Total number of spatial degrees of freedom for fine grid
            'diameter': np.sqrt(2)/N,                     # Max diameter of elements in coarse mesh
            'fine_diameter': np.sqrt(2)/fine_N,           # Max diameter of elements in fine mesh
            'par_dim': par_dim,                           # Dimension of parameter space (e.g., number of parameters to infer)
            'output_dim': 1,                              # Dimension of model output (e.g., scalar output per time step)
        },
        'problem_parameter': {
            'N': N,                                       # Grid size used in the PDE problem
            'contrast_parameter': 2,                      # Material contrast in the diffusion coefficient
            'parameter_location': 'reaction',             # Location in PDE where parameter acts (e.g., in reaction term)
            'boundary_conditions': 'dirichlet',           # Type of boundary conditions applied (fixed value)
            'exact_parameter': 'Kirchner',                # Shape or distribution of true parameter (used for testing)
            'time_factor': 'sinus',                       # Time dependence type of source or parameter (e.g., sinusoidal)
            'T_final': T_final,                           # Final simulation time
        },
        'model_parameter': {
            'name': 'reaction_FOM',                       # Name of the model, e.g., Full Order Model for reaction-diffusion
            'problem_type': None,                         # Problem type, will be set by 'build_InstationaryModelIP'
            'T_initial': T_initial,                       # Start time of the simulation
            'T_final': T_final,                           # End time of the simulation
            'delta_t': delta_t,                           # Time step size
            'noise_percentage': None,                     # Relative noise level, will be set by 'build_InstationaryModelIP'
            'noise_level': noise_level,                   # Absolute noise magnitude added to data
            'q_circ': q_circ,                             # Backgroundlevel for the parameter
            'q_exact_function': None,                     # Exact parameter as function, will be set by 'build_InstationaryModelIP'
            'q_exact': None,                              # Exact parameter values, will be set by 'build_InstationaryModelIP'
            'q_time_dep': q_time_dep,                     # Whether parameter is time-dependent (bool)
            'riesz_rep_grad': True,                       # Use Riesz representative for gradient in optimization
            'bounds': bounds,                             # Bounds on parameter values (e.g., for optimization)
            'parameters': None,                           # List of the parameters use.
            'products': {                                 # Inner products used in the problem
                'prod_H': 'l2',                           # Product on H_h
                'prod_Q': 'l2',                           # Product on Q_h
                'prod_V': 'h1_0_semi',                    # Product on V_h
                'prod_C': 'l2',                           # Product on C_h
                'bochner_prod_Q': 'bochner_l2',           # Product on Q_h^K
                'bochner_prod_V': 'bochner_h1_0_semi'     # Product on V_h^K
            },
            'observation_operator': {
                'name': 'identity',                       # Type of observation operator (e.g., identity = full state observed)
            }
        }
    }

    FOM, _, _ = build_InstationaryModelIP(setup, logger)
    q_exact = FOM.setup['model_parameter']['q_exact']
    q_start = q_circ

    # FOM.visualizer.visualize(q_exact)
    # import sys
    # sys.exit()
    # np.random.seed(42)
    # q_start = np.random.random(q_exact.to_numpy().shape)

    optimizer_parameter = {
        'q_0': q_start,                                              # Initial guess for the parameter to be optimized
        'alpha_0': 1e-5,                                             # Initial regularization parameter (data fidelity vs. regularization)
        'tol': 1e-9,                                                 # Absolute convergence tolerance for optimization
        'tau': 3.5,                                                  # Relative (to the noise) convergence tolerance for optimization
        'noise_level': setup['model_parameter']['noise_level'],      # Noise level in observed data (from model setup)
        'theta': 0.4,                                                # Lower bound for step acceptance condition
        'Theta': 1.95,                                               # Upper bound for step acceptance condition
        'tau_tilde': 3.5,                                            # Relative (to the noise) convergence tolerance for optimization inside the trust region
        #####################
        'i_max': 75,                                                 # Max number of outer optimization iterations
        'reg_loop_max': 10,                                          # Max number of regularization updates per iteration
        'i_max_inner': 10,                                           # Max number of inner iterations
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
            'parameter_HaPOD_tol': 1e-12,                            # Tolerance for parameter basis POD
            'state_strategy': 'snapshot_HaPOD',                      # Enrichment strategy for state basis
            'state_HaPOD_tol': 1e-9                                  # Tolerance for state basis POD
        },
        #####################
        'use_cached_operators': True,                                # Reuse previously assembled operators to save computation
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
    logger.debug("Differnce to q_exact:")
    logger.debug("L^inf") 
    delta_q = q_est - q_exact
    logger.debug(f"  {np.max(np.abs(delta_q.to_numpy())):3.4e}")
    
    if q_time_dep:
        norm_delta_q = np.sqrt(FOM.products['bochner_prod_Q'].apply2(delta_q, delta_q))[0,0]
        norm_q_exact = np.sqrt(FOM.products['bochner_prod_Q'].apply2(q_exact, q_exact))[0,0]
    else:
        norm_delta_q = np.sqrt(FOM.products['prod_Q'].apply2(delta_q, delta_q))[0,0]
        norm_q_exact = np.sqrt(FOM.products['prod_Q'].apply2(q_exact, q_exact))[0,0]
    
    logger.debug(f"  Absolute error: {norm_delta_q:3.4e}")
    logger.debug(f"  Relative error: {norm_delta_q / norm_q_exact * 100:3.4}%.")

if __name__ == '__main__':
    main()
