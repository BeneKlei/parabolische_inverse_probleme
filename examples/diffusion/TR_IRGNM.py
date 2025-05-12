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
    N = 300
    #N = 100
    #N = 30
    par_dim = (N+1)**2
    fine_N = 2 * N


    T_initial = 0
    T_final = 1
    # TODO Here is a Bug
    nt = 50
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
        'dims' : {
            'N': N,
            'nt': nt,
            'fine_N': fine_N,
            'state_dim': (N+1)**2,
            'fine_state_dim': (fine_N+1)**2,
            'diameter': np.sqrt(2)/N,
            'fine_diameter': np.sqrt(2)/fine_N,
            'par_dim': par_dim,
            'output_dim': 1,                                                                                                                                                                         # options to preassemble affine components or not
        },
        'problem_parameter' : {
            'N': N,
            'contrast_parameter' : 2,
            'parameter_location' : 'diffusion',
            'boundary_conditions' : 'dirichlet',
            'exact_parameter' : 'PacMan',
            #'time_factor' : 'constant',
            'time_factor' : 'sinus',
            'T_final' : T_final,
        },
        'model_parameter' : {
            'name' : 'diffusion_TR', 
            'problem_type' : None,
            'T_initial' : T_initial,
            'T_final' : T_final,
            'delta_t' : delta_t,
            'noise_percentage' : None,
            'noise_level' : noise_level,
            'q_circ' : q_circ, 
            'q_exact_function' : None,
            'q_exact' : None,
            'q_time_dep' : q_time_dep,
            'riesz_rep_grad' : True,
            'bounds' : bounds,
            'parameters' : None,
            'products' : {
                'prod_H' : 'l2',
                'prod_Q' : 'h1',
                'prod_V' : 'h1_0_semi',
                'prod_C' : 'l2',
                'bochner_prod_Q' : 'bochner_h1',
                'bochner_prod_V' : 'bochner_h1_0_semi'
            },
            'observation_operator' : {
                'name' : 'identity',
            }
            # 'observation_operator' : {
            #     'name' : 'RoI',
            #     'RoI' : np.array([[0.0,0.5], [0.0,0.5]])
            # }
        }
    }

    FOM, _, _ = build_InstationaryModelIP(setup, logger)
    q_exact = FOM.setup['model_parameter']['q_exact']
    q_start = q_circ

    optimizer_parameter = {
        'q_0' : q_start,
        'alpha_0' : 1e-5,
        'tol' : 1e-9,
        'tau' : 3.5,
        'noise_level' : setup['model_parameter']['noise_level'],
        'theta' : 0.4,
        'Theta' : 1.95,
        'tau_tilde' : 3.5,
        #####################
        'i_max' : 75,
        'reg_loop_max' : 10,
        'i_max_inner' : 10,
        'agc_armijo_max_iter' : 100,
        #'TR_armijo_max_iter' : 10,
        'TR_armijo_max_iter' : 5,
        #####################
        'lin_solver_parms' : {
            'method' : 'gd',
            'max_iter' : 1e4,
            'lin_solver_tol' : 1e-10,
            'inital_step_size' : 1
        },
        # 'lin_solver_parms' : {
        #     'method' : 'BiCGSTAB',
        #     'rtol' : 1e-12,
        #     'atol' : 1e-12,
        #     'maxiter' : 1e3
        # },
        'enrichment' : {
            'parameter_strategy' : 'snapshot_HaPOD',
            'parameter_HaPOD_tol': 1e-12,
            'state_strategy' : 'snapshot_HaPOD',
            'state_HaPOD_tol': 1e-9,
        },
        'use_cached_operators' : True,
        'dump_every_nth_loop' : 2,        
        #####################
        'eta0' : 1e-1,
        'kappa_arm' : 1e-12,
        'beta_1' : 0.95,
        'beta_2' : 3/4,
        'beta_3' : 0.5,
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
