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

    N = 100
    #N = 10
    par_dim = (N+1)**2
    fine_N = 2 * N

    T_initial = 0
    T_final = 1
    # TODO Here is a Bug
    nt = 50
    delta_t = (T_final - T_initial) / nt
    #q_time_dep = False
    q_time_dep = True

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
            'parameter_location' : 'diffusion',
            'boundary_conditions' : 'dirichlet',
            'exact_parameter' : 'PacMan',
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
    #q_exact = FOM.setup['model_parameter']['q_exact']
    
    q = FOM.Q.make_array(q_circ)
    u = FOM.solve_state(q)
    p = FOM.solve_adjoint(q, u)
    J = FOM.objective(u)
    nabla_J = FOM.gradient(u, p)
    d = q.copy()


    # reductor.extend_basis(
    #     U = nabla_J,
    #     basis = 'parameter_basis'
    # )
    # QrFOM = reductor.reduce()
    # d = FOM.Q.make_array(q_circ)

    # q_r = reductor.project_vectorarray(q, 'parameter_basis')
    # q_r = QrFOM.Q.make_array(q_r)
    # d_r = reductor.project_vectorarray(d, 'parameter_basis')
    # d_r = QrFOM.Q.make_array(d_r)

    #########################################################################################

    import cProfile
    import os

    if os.path.exists('profiling_results.prof'):
        os.remove('profiling_results.prof')
        
    pr = cProfile.Profile()

    #QrFOM.compute_linearized_gradient(q_r, d_r, alpha=1)
    pr.enable()
    FOM.compute_linearized_gradient(q, d, alpha=1)
    # import sys
    # sys.exit()
    pr.disable()
    pr.dump_stats('profiling_results_FOM_q_dep.prof')
    #os.system('snakeviz profiling_results.prof')
        

if __name__ == '__main__':
    main()
