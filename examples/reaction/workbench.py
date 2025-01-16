import numpy as np
import logging

from pymor.basic import *

from RBInvParam.model import InstationaryModelIP
from RBInvParam.problems.problems import whole_problem
from RBInvParam.discretizer import discretize_instationary_IP
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.reductor import InstationaryModelIPReductor
from RBInvParam.problems.problems import build_InstationaryModelIP


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

N = 10
#N = 100
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
        'N' : N,
        'parameter_location' : 'reaction',
        'boundary_conditions' : 'dirichlet',
        'exact_parameter' : 'Kirchner',
        'state_space_product' : 'h1' 
    },
    'model_parameter' : {
        'name' : 'reaction_FOM', 
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
}

setup, FOM = build_InstationaryModelIP(setup, logger)
q_exact = setup['model_parameter']['q_exact']



reductor = InstationaryModelIPReductor(FOM)

#q = FOM.Q.make_array(q_circ)
#u = FOM.solve_state(q)
# p = FOM.solve_adjoint(q, u)
# J = FOM.objective(u)
# nabla_J = FOM.gradient(u, p)

reductor.extend_basis(
    U = q_exact,
    basis = 'parameter_basis'
)

QrFOM = reductor.reduce()
#d = FOM.Q.make_array(q_exact)

q_r = reductor.project_vectorarray(q_exact, 'parameter_basis')
q_r = QrFOM.Q.make_array(q_r)
# d_r = reductor.project_vectorarray(d, 'parameter_basis')
# d_r = QrFOM.Q.make_array(d_r)

u = FOM.solve_state(q_exact)
u_r = QrFOM.solve_state(q_r)

p = FOM.solve_adjoint(q_exact, u)
p_r = QrFOM.solve_adjoint(q_r, u_r)


delta_u = u - u_r
print(np.sqrt(FOM.products['bochner_prod_V'].apply2(delta_u, delta_u)))

delta_p = p - p_r
print(np.sqrt(FOM.products['bochner_prod_V'].apply2(delta_p, delta_p)))

print("###########################################")
print(QrFOM.estimate_state_error(
    q = q_r, 
    u = u_r))
print(QrFOM.estimate_adjoint_error(
    q = q_r, 
    u = u_r, 
    p = p_r))

#########################################################################################