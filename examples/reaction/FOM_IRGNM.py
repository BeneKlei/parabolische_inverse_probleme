import numpy as np
from pathlib import Path
import logging

from pymor.basic import *

from RBInvParam.optimizer import FOMOptimizer
from RBInvParam.utils.io import save_dict_to_pkl
from RBInvParam.utils.logger import get_default_logger

from RBInvParam.problems.problems import build_InstationaryModelIP
# TODO
# - Find better way to handle time independ parameter

#########################################################################################''

logger = get_default_logger(logfile_path='./logs/FOM_IRGNM.log', use_timestemp=True)
logger.setLevel(logging.DEBUG)

set_log_levels({
    'pymor' : 'WARN'
})

set_defaults({})

#########################################################################################''

def main():

    #N = 100
    N = 10
    par_dim = (N+1)**2
    fine_N = 2 * N

    T_initial = 0
    T_final = 1
    # TODO Here is a Bug
    nt = 50
    delta_t = (T_final - T_initial) / nt
    q_time_dep = False
    #q_time_dep = True

    #noise_level = 1e-7
    noise_level = 0.0
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
            'parameter_location' : 'reaction',
            'boundary_conditions' : 'dirichlet',
            'exact_parameter' : 'Kirchner',
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
            'bounds' : bounds,
            'parameters' : None,
            'products' : {
                'prod_H' : 'l2',
                'prod_Q' : 'l2',
                'prod_V' : 'h1',
                'prod_C' : 'l2',
                'bochner_prod_Q' : 'bochner_l2',
                'bochner_prod_V' : 'bochner_h1'
            }
        }
    }

    FOM = build_InstationaryModelIP(setup, logger)
    q_exact = FOM.setup['model_parameter']['q_exact']

    # if q_time_dep:
    #     q_start = 0*np.ones((nt, par_dim))
    # else:
    #     q_start = 0*np.ones((1, par_dim))
    q_start = q_circ

    optimizer_parameter = {
        'noise_level' : FOM.setup['model_parameter']['noise_level'],
        'tau' : 3.5,
        'tol' : 1e-8,
        'q_0' : q_start,
        'alpha_0' : 1e-5,
        'i_max' : 50,
        'i_max_inner' : 2,
        'reg_loop_max' : 10,
        'theta' : 0.25,
        'Theta' : 0.75,
    }

    optimizer = FOMOptimizer(
        FOM = FOM,
        optimizer_parameter = optimizer_parameter,
        logger = logger
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
