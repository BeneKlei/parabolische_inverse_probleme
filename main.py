import numpy as np
from pymor.basic import *
from problems import whole_problem
from discretizer import discretize_instationary_IP
from pymor.parameters.base import ParameterSpace

# general options           
N = 10                                                                         # FE Dofs = (N+1)^2                                                
noise_level = 1e-5        

print('Construct problem..')                                                     
analytical_problem, q_exact, N, problem_type, exact_analytical_problem, energy_problem = whole_problem(
                                                        N = N,
                                                        parameter_location = 'reaction',
                                                        boundary_conditions = 'dirichlet',
                                                        exact_parameter = 'Kirchner',
                                                       )

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
    'noise_percentage' : None,
    'noise_level' : 0.05,
    'q_circ' : 3*np.ones((nt, dims['par_dim'])), 
    'q_exact' : q_exact,
    'bounds' : bounds,
    #'parameter_space' : ParameterSpace(analytical_problem.parameters, bounds) 
}

print('Discretizing problem...')                                                
# discretize analytical problem to obtain inverse problem fom
fom_IP, fom_IP_data = discretize_instationary_IP(analytical_problem,
                                                 model_parameter,
                                                 dims
                                               ) 

