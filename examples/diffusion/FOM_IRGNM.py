import numpy as np
import logging
import os

from pathlib import Path
from datetime import datetime

from pymor.basic import *

from RBInvParam.optimizer import FOMOptimizer
from RBInvParam.utils.io import save_dict_to_pkl
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.problems.problems import build_InstationaryModelIP

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

    N = 100
    #N = 30
    par_dim = (N+1)**2
    fine_N = 2 * N

    T_initial = 0
    T_final = 1
    nt = 50
    delta_t = (T_final - T_initial) / nt
    #q_time_dep = False
    q_time_dep = True

    noise_level = 1e-5
   # noise_level = 0.0

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
            'parameter_location': 'diffusion',            # Location in PDE where parameter acts (e.g., in diffusion term)
            'boundary_conditions': 'dirichlet',           # Type of boundary conditions applied (fixed value)
            'exact_parameter': 'PacMan',                  # Shape or distribution of true parameter (used for testing)
            'time_factor': 'sinus',                       # Time dependence type of source or parameter (e.g., sinusoidal)
            'T_final': T_final,                           # Final simulation time
        },
        'model_parameter': {
            'name': 'diffusion_FOM',                      # Name of the model, e.g., Full Order Model for reaction-diffusion
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
                'prod_Q': 'h1',                           # Product on Q_h
                'prod_V': 'h1_0_semi',                    # Product on V_h
                'prod_C': 'l2',                           # Product on C_h
                'bochner_prod_Q': 'bochner_h1',           # Product on Q_h^K
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

    optimizer_parameter = {
        'q_0': q_start,                                          # Initial guess for the parameter to be optimized
        'alpha_0': 1e-5,                                         # Initial regularization parameter
        'tol': 1e-11,                                            # Absolute convergence tolerance for optimization
        'tau': 3.5,                                              # Relative (to the noise) convergence tolerance for optimization
        'noise_level': setup['model_parameter']['noise_level'],  # Noise level in observed data (from model setup)
        'theta': 0.4,                                            # Lower tolerance for the direction acceptance condition
        'Theta': 0.95,                                           # Upper tolerance for the direction acceptance condition
        #####################
        'i_max': 35,                                             # Maximum number of outer optimization iterations
        'reg_loop_max': 10,                                      # Maximum number of regularization updates per step
        'i_max_inner': 10,                                       # Maximum number of inner iterations
        ####################
        'lin_solver_parms': {
            'method' : 'gd',                                     # Method for solving linear systems (e.g., gradient descent)
            'max_iter': 1e4,                                     # Max iterations for the linear solver
            'lin_solver_tol': 1e-12,                             # Tolerance for convergence in the linear solver
            'inital_step_size': 1                                # Initial step size for iterative solvers (if applicable)
        },
        'use_cached_operators': True,                            # Whether to reuse assembled operators (improves speed if True)
        'dump_every_nth_loop': 2                                 # Dump intermediate results every n optimization iterations
    }

    optimizer = FOMOptimizer(
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
