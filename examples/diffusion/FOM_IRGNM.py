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
# TODO
# - Find better way to handle time independ parameter

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
    # TODO Here is a Bug
    nt = 500
    delta_t = (T_final - T_initial) / nt
    #q_time_dep = False
    q_time_dep = True

    noise_level = 1e-5
   # noise_level = 0.0
    bounds = [0.001*np.ones((par_dim,)), 10e2*np.ones((par_dim,))]

    assert T_final > T_initial
    if q_time_dep:
        q_circ = 3*np.ones((nt, par_dim))
    else:
        q_circ = 3*np.ones((1, par_dim))

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
            'time_factor' : 'sinus',
            'T_final' : T_final,
        },
        'model_parameter' : {
            'name' : 'reaction_FOM', 
            'problem_type' : None,
            'T_initial' : T_initial,
            'T_final' : T_final,
            'delta_t' : delta_t,
            'noise_percentage' : None,
            'noise_level' : noise_level,
            'q_circ' : q_circ, 
            'q_exact' : None,
            'q_time_dep' : q_time_dep,
            'riesz_rep_grad' : True,
            'bounds' : bounds,
            'parameters' : None,
            'products' : {
                'prod_H' : 'l2',
                'prod_Q' : 'l2',
                'prod_V' : 'h1_0_semi',
                'prod_C' : 'l2',
                'bochner_prod_Q' : 'bochner_l2',
                'bochner_prod_V' : 'bochner_h1'
            },
            'observation_operator' : {
                'name' : 'identity',
            }
        }
    }

    FOM, _, _ = build_InstationaryModelIP(setup, logger)
    q_exact = FOM.setup['model_parameter']['q_exact']
    q_start = q_circ

    optimizer_parameter = {
        'q_0' : q_start,
        'alpha_0' : 1e-5,
        'tol' : 1e-11,
        'tau' : 3.5,
        'noise_level' : setup['model_parameter']['noise_level'],
        'theta' : 0.4,
        #'Theta' : 0.75,
        'Theta' : 0.95,
        #####################
        'i_max' : 35,
        'reg_loop_max' : 10,
        'i_max_inner' : 2,
        ####################
        'lin_solver_parms' : {
            'lin_solver_max_iter' : 1e4,
            'lin_solver_tol' : 1e-12,
            'lin_solver_inital_step_size' : 1
        },
        'use_cached_operators' : True
    }

    optimizer = FOMOptimizer(
        FOM = FOM,
        optimizer_parameter = optimizer_parameter,
        logger = logger,
        save_path=save_path
    )
    q_est = optimizer.solve()

    
    FOM.visualizer.visualize(q_est, title="q_est")
    FOM.visualizer.visualize(q_exact, title="q_exact")
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


    save_path = Path(f"./dumps/FOM_IRGNM_{N}.pkl")
    logger.debug(f"Save statistics to {save_path}")

    data = {
        'setup' : FOM.setup,
        'optimizer_statistics' : optimizer.statistics
    }

    save_dict_to_pkl(path=save_path, data=data, use_timestamp=False)

if __name__ == '__main__':
    main()
