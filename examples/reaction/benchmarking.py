import numpy as np
import logging

from pymor.basic import *

from RBInvParam.model import InstationaryModelIP
from RBInvParam.problems.problems import whole_problem
from RBInvParam.discretizer import discretize_instationary_IP
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.reductor import InstationaryModelIPReductor


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

#N = 10
N = 100
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
    model_parameter = model_parameter,
    name='reaction_FOM'
)
reductor = InstationaryModelIPReductor(FOM)

q = FOM.Q.make_array(q_circ)
u = FOM.solve_state(q)
p = FOM.solve_adjoint(q, u)
J = FOM.objective(u)
nabla_J = FOM.gradient(u, p)

reductor.extend_basis(
    U = nabla_J,
    basis = 'parameter_basis'
)
Q_r_ROM = reductor.reduce()
d = FOM.Q.make_array(q_circ)

q_r = reductor.project_vectorarray(q, 'parameter_basis')
q_r = Q_r_ROM.Q.make_array(q_r)
d_r = reductor.project_vectorarray(d, 'parameter_basis')
d_r = Q_r_ROM.Q.make_array(d_r)

#########################################################################################

import cProfile
import os

if os.path.exists('profiling_results.prof'):
    os.remove('profiling_results.prof')
    
pr = cProfile.Profile()


pr.enable()
#FOM.compute_linearized_gradient(q, d, alpha=1)
Q_r_ROM.compute_linearized_gradient(q_r, d_r, alpha=1)
pr.disable()

pr.dump_stats('profiling_results.prof')

os.system('snakeviz profiling_results.prof')