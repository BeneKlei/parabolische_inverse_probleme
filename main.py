import numpy as np
import logging
import sys
sys.path.append('./')

from pymor.basic import *
from pymor.basic import *
import numpy as np
from problems import whole_problem
from discretizer import discretize_instationary_IP
from pymor.parameters.base import ParameterSpace

from model import InstationaryModelIP
from optimizer import FOMOptimizer

set_log_levels({
    'pymor' : 'WARN'
})

set_defaults({
    # 'pymor.algorithms.gram_schmidt.gram_schmidt.atol' : 1e-15,
    # 'pymor.algorithms.gram_schmidt.gram_schmidt.rtol' : 1e-15,
})

N = 10                                                                      # FE Dofs = (N+1)^2                                                
noise_level = 1e-5        
nt = 10
fine_N = 2 * N

dims = {
    'N': N,
    'nt': nt,
    'fine_N': fine_N,
    'state_dim': (N+1)**2,
    'fine_state_dim': (fine_N+1)**2,
    'diameter': np.sqrt(2)/N,
    'fine_diameter': np.sqrt(2)/fine_N,
    'par_dim': (N+1)**2,
    'output_dim': 1,                                                                                                                                                                         # options to preassemble affine components or not
}

bounds = [0.001*np.ones((dims['par_dim'],)), 10e2*np.ones((dims['par_dim'],))]
q_circ = 3*np.ones((nt, dims['par_dim']))

model_parameter = {
    'T_initial' : 0,
    'T_final' : 1,
    'noise_percentage' : None,
    'noise_level' : 0.00,
    'q_circ' : q_circ, 
    'q_exact' : None,
    'bounds' : bounds,
    #'parameter_space' : ParameterSpace(analytical_problem.parameters, bounds) 
    'parameters' : None
}

optimizer_parameter = {
    'noise_level' : model_parameter['noise_level'],
    'tau' : 1e-10,
    'tol' : 1e-10,
    'q_0' : q_circ,
    'alpha_0' : 1e-6,
    'i_max' : 50,
    'reg_loop_max' : 50,
    'theta' : 0.25,
    'Theta' : 0.75
}

print('Construct problem..')                                                     
analytical_problem, q_exact, N, problem_type, exact_analytical_problem, energy_problem = whole_problem(
                                                        N = N,
                                                        parameter_location = 'reaction',
                                                        boundary_conditions = 'dirichlet',
                                                        exact_parameter = 'Kirchner',
                                                       )

model_parameter['q_exact'] = q_exact
model_parameter['parameters'] = analytical_problem.parameters

print('Discretizing problem...')                                                
# discretize analytical problem to obtain inverse problem fom
building_blocks = discretize_instationary_IP(analytical_problem,
                            model_parameter,
                            dims, 
                            problem_type
                        ) 

FOM = InstationaryModelIP(                 
    *building_blocks,
    dims = dims,
    model_parameter = model_parameter
)

optimizer = FOMOptimizer(
    FOM = FOM,
    optimizer_parameter = optimizer_parameter
)
optimizer.logger.setLevel(logging.INFO)

optimizer.solve()