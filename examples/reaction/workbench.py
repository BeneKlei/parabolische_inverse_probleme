import numpy as np
import logging
import os
from pathlib import Path
from datetime import datetime

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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = Path('./dumps') / (timestamp + '_workbench')
os.mkdir(save_path)
logfile_path= save_path / 'workbench.log'

logger = get_default_logger(logger_name='workbench',
                            logfile_path=logfile_path, 
                            use_timestemp=False)
logger.setLevel(logging.DEBUG)

#########################################################################################''

set_log_levels({
    'pymor' : 'WARN'
})

set_defaults({})

#########################################################################################''

N = 30
#N = 100
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
        'riesz_rep_grad' : True,
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
reductor = InstationaryModelIPReductor(FOM)

q = q_circ
q = FOM.Q.make_array(q)
u = FOM.solve_state(q, use_cached_operators=True)
p = FOM.solve_adjoint(q, u, use_cached_operators=True)
J = FOM.objective(u)
nabla_J = FOM.gradient(u, p, q, use_cached_operators=True)

print(nabla_J)

import sys
sys.exit()

reductor.extend_basis(
    U = q,
    basis = 'parameter_basis'
)

state_shapshots = FOM.V.empty()
state_shapshots.append(u)
state_shapshots.append(p)

# self.logger.debug(f"Performing HaPOD on state snapshots.")
# state_shapshots, _ = self._HaPOD(shapshots=state_shapshots, 
#                                     basis='state_basis',
#                                     product=self.FOM.products['prod_V'])

reductor.extend_basis(
        U = state_shapshots,
        basis = 'state_basis'
)

QrVrROM = reductor.reduce()

#QrFOM = reductor.reduce()
#d = FOM.Q.make_array(q_exact)

q_r = reductor.project_vectorarray(q, 'parameter_basis')
q_r = QrVrROM.Q.make_array(q_r)


print(FOM.compute_objective(q))
print(QrVrROM.compute_objective(q_r))

# d_r = reductor.project_vectorarray(d, 'parameter_basis')
# d_r = QrFOM.Q.make_array(d_r)

u = FOM.solve_state(q)
u_r = QrVrROM.solve_state(q_r)

p = FOM.solve_adjoint(q, u)
p_r = QrVrROM.solve_adjoint(q_r, u_r)

print(QrVrROM.estimate_objective_error(q_r, u_r, p_r))
import sys
sys.exit()


delta_u = u - u_r
print(np.sqrt(FOM.products['bochner_prod_V'].apply2(delta_u, delta_u)[0,0]))

delta_p = p - p_r
print(np.sqrt(FOM.products['bochner_prod_V'].apply2(delta_p, delta_p)[0,0]))

print("###########################################")
print(QrFOM.estimate_state_error(
    q = q_r, 
    u = u_r))
# print(QrFOM.estimate_adjoint_error(
#     q = q_r, 
#     u = u_r, 
#     p = p_r))

#########################################################################################