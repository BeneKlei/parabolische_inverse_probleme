import numpy as np
from pymor.basic import *
from problems import whole_problem
from discretizer import discretize_instationary_IP
from pymor.parameters.base import ParameterSpace

from model import InstationaryModelIP

# general options           
N = 10                                                                         # FE Dofs = (N+1)^2                                                
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

model_parameter = {
    'T_initial' : 0,
    'T_final' : 1,
    'noise_percentage' : None,
    'noise_level' : 0.05,
    'q_circ' : 3*np.ones((nt, dims['par_dim'])), 
    'q_exact' : None,
    'bounds' : bounds,
    #'parameter_space' : ParameterSpace(analytical_problem.parameters, bounds) 
    'parameters' : None
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

model = InstationaryModelIP(
    *building_blocks,
    dims = dims,
    model_parameter = model_parameter
)

q_exact = []
for _ in range(dims['nt'] + 1):
    q_exact.append(model_parameter['q_exact'])
q_exact = np.array(q_exact)

U = model.solve_state(q_exact)
P = model.solve_adjoint(q_exact, U)