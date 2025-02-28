import numpy as np
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Union, Tuple


from pymor.basic import *
from pymor.vectorarrays.interface import VectorArray
from pymor.algorithms.hapod import inc_vectorarray_hapod
from pymor.operators.interface import Operator

from RBInvParam.reductor import InstationaryModelIPReductor
from RBInvParam.utils.io import save_dict_to_pkl
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.problems.problems import build_InstationaryModelIP


def _HaPOD(reductor, 
           shapshots: VectorArray, 
           basis: str,
           product: Operator,
           eps: float = 1e-16) -> Tuple[VectorArray, np.array]:
        
    if len(reductor.bases[basis]) != 0:
        projected_shapshots = reductor.bases[basis].lincomb(
            reductor.project_vectorarray(shapshots, basis=basis)
        )
        shapshots.axpy(-1,projected_shapshots)
            
    shapshots, svals, _ = \
    inc_vectorarray_hapod(steps=len(shapshots)/5, 
                            U=shapshots, 
                            eps=eps,
                            omega=0.1,                
                            product=product)


    return shapshots, svals


# TODO
# - Find better way to handle time independ parameter

#########################################################################################''

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = Path('./dumps') / (timestamp + '_benchmarking')
os.mkdir(save_path)
logfile_path= save_path / 'benchmarking.log'

logger = get_default_logger(logger_name='benchmarking',
                            logfile_path=logfile_path, 
                            use_timestemp=False)
logger.setLevel(logging.DEBUG)

#########################################################################################''

set_log_levels({
    'pymor' : 'WARN'
})

set_defaults({})


#########################################################################################''

#N = 30
#N = 100
N = 300
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
        'riesz_rep_grad' : True,
        'bounds' : bounds,
        'parameters' : None,
        'products' : {
            'prod_H' : 'l2',
            'prod_Q' : 'h1',
            'prod_V' : 'h1',
            'prod_C' : 'l2',
            'bochner_prod_Q' : 'bochner_h1',
            'bochner_prod_V' : 'bochner_h1'
        }
    }
}


FOM = build_InstationaryModelIP(setup, logger)
q_exact = FOM.setup['model_parameter']['q_exact']
u = FOM.solve_state(q_exact)


logger.info(f"Dumping model setup to {save_path / 'setup.pkl'}.")
save_dict_to_pkl(path=save_path / 'setup.pkl', 
                    data = setup,
                    use_timestamp=False)
reductor = InstationaryModelIPReductor(FOM)

q = FOM.Q.make_array(q_circ)
u = FOM.solve_state(q)
p = FOM.solve_adjoint(q, u)
J = FOM.objective(u)
nabla_J = FOM.gradient(u, p, q)

reductor.extend_basis(
    U = nabla_J,
    basis = 'parameter_basis'
)

# QrFOM = reductor.reduce()
# d = FOM.Q.make_array(q_circ)

# q_r = reductor.project_vectorarray(q, 'parameter_basis')
# q_r = QrFOM.Q.make_array(q_r)
# d_r = reductor.project_vectorarray(d, 'parameter_basis')
# d_r = QrFOM.Q.make_array(d_r)

state_shapshots = FOM.V.empty()
state_shapshots.append(u)
state_shapshots.append(p)

state_shapshots, _ = _HaPOD(reductor = reductor,
                            shapshots=state_shapshots, 
                            basis='state_basis',
                            product=FOM.products['prod_V'])
        
reductor.extend_basis(
        U = state_shapshots,
        basis = 'state_basis'
)

QrVrROM = reductor.reduce()
d = FOM.Q.make_array(q_circ)

q_r = reductor.project_vectorarray(q, 'parameter_basis')
q_r = QrVrROM.Q.make_array(q_r)
d_r = reductor.project_vectorarray(d, 'parameter_basis')
d_r = QrVrROM.Q.make_array(d_r)

u_r = QrVrROM.solve_state(q_r)
p_r = QrVrROM.solve_adjoint(q_r, u_r)

#########################################################################################

import cProfile
import os

if os.path.exists('profiling_results.prof'):
    os.remove('profiling_results.prof')
    
pr = cProfile.Profile()

#QrFOM.compute_linearized_gradient(q_r, d_r, alpha=1)
pr.enable()
print("Starting")
#QrVrROM.estimate_objective_error(q_r, u_r, p_r, use_cached_operators=False)
#FOM.compute_linearized_gradient(q, d, alpha=1, use_cached_operators=True)
#QrFOM.compute_linearized_gradient(q_r, d_r, alpha=1, use_cached_operators=True)
FOM.compute_linearized_gradient(q, d, alpha=1, use_cached_operators=True)
# import sys
# sys.exit()
pr.disable()

pr.dump_stats('profiling_results_QrVrROM_cached_.prof')

#os.system('snakeviz profiling_results.prof')