import numpy as np
from pathlib import Path
import logging

from pymor.basic import *
from pymor.parameters.base import ParameterSpace

from RBInvParam.model import InstationaryModelIP
from RBInvParam.optimizer import FOMOptimizer
from RBInvParam.problems.problems import whole_problem
from RBInvParam.discretizer import discretize_instationary_IP
from RBInvParam.utils.io import save_dict_to_pkl
from RBInvParam.utils.logger import get_default_logger

from RBInvParam.gradient_descent import gradient_descent_non_linearized_problem

# TODO
# - Find better way to handle time independ parameter

#########################################################################################''

logger = get_default_logger(logfile_path='./logs/log.log', use_timestemp=True)
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

    noise_level = 1e-5
    bounds = [0.001*np.ones((par_dim,)), 10e2*np.ones((par_dim,))]

    assert T_final > T_initial
    if q_time_dep:
        q_circ = 3*np.ones((nt, par_dim))
    else:
        q_circ = 3*np.ones((1, par_dim))


    dims = {
        'N': N,
        'nt': nt,
        'fine_N': fine_N,
        'state_dim': (N+1)**2,
        'fine_state_dim': (fine_N+1)**2,
        'diameter': np.sqrt(2)/N,
        'fine_diameter': np.sqrt(2)/fine_N,
        'par_dim': par_dim,
        'output_dim': 1,                                                                                                                                                                         # options to preassemble affine components or not
    }

    problem_parameter = {
        'N' : N,
        'parameter_location' : 'reaction',
        'boundary_conditions' : 'dirichlet',
        'exact_parameter' : 'Kirchner',
    }

    model_parameter = {
        'T_initial' : T_initial,
        'T_final' : T_final,
        'delta_t' : delta_t,
        'noise_percentage' : None,
        'noise_level' : noise_level,
        'q_circ' : q_circ, 
        'q_exact' : None,
        'q_time_dep' : q_time_dep,
        'bounds' : bounds,
        'parameters' : None
    }


    logger.debug('Construct problem..')                                                     
    analytical_problem, q_exact, N, problem_type, exact_analytical_problem, energy_problem = whole_problem(**problem_parameter)

    model_parameter['parameters'] = analytical_problem.parameters
    if q_time_dep:                                                 
        model_parameter['q_exact'] = np.array([q_exact for _ in range(dims['nt'])])
    else:
        model_parameter['q_exact'] = np.array([q_exact])

    logger.debug('Discretizing problem...')                                                
    building_blocks = discretize_instationary_IP(analytical_problem,model_parameter,dims, problem_type) 


    FOM = InstationaryModelIP(                 
        *building_blocks,
        dims = dims,
        model_parameter = model_parameter
    )

    # if q_time_dep:
    #     q_start = 0*np.ones((nt, par_dim))
    # else:
    #     q_start = 0*np.ones((1, par_dim))
    q_start = q_circ

    optimizer_parameter = {
        'noise_level' : model_parameter['noise_level'],
        'tau' : 1e-4,
        'tol' : 1e-13,
        'q_0' : q_start,
        'alpha_0' : 1e-5,
        #'alpha_0' : 1e-1,
        'i_max' : 50,
        'reg_loop_max' : 50,
        'theta' : 0.25,
        'Theta' : 0.75
    }

    optimizer = FOMOptimizer(
        FOM = FOM,
        optimizer_parameter = optimizer_parameter,
        logger = logger
    )
    q_est = optimizer.solve()

    q_exact = FOM.Q.make_array(model_parameter['q_exact'])
    FOM.visualizer.visualize(q_est, title="q_est")
    FOM.visualizer.visualize(q_exact, title="q_exact")
    logger.debug("Differnce to q_exact:")
    logger.debug("L^inf") 
    logger.debug(f"{np.max(np.abs((q_est - q_exact).to_numpy())):3.4e}")
    logger.debug("Q-Norm") 
    norm_delta_q = np.sqrt(FOM.products['bochner_prod_Q'](q_est - q_exact, q_est - q_exact))
    norm_q_exact = np.sqrt(FOM.products['bochner_prod_Q'](q_exact, q_exact))
    logger.debug(f"Absolute error: {norm_delta_q:3.4e}")
    logger.debug(f"Relative error: {norm_delta_q / norm_q_exact * 100:3.4}%.")

    save_path = Path("./dumps/test.pkl")
    logger.debug(f"Save statistics to {save_path}")

    data = {
        'dims' : dims,
        'model_parameter' : model_parameter,
        'optimizer_parameter' : optimizer_parameter,
        'optimizer_statistics' : optimizer.statistics
    }

    save_dict_to_pkl(path=save_path, data=data)

if __name__ == '__main__':
    main()
