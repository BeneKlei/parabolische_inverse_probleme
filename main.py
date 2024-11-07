import numpy as np
from pymor.basic import *
from problems import whole_problem
from discretizer import discretize_instationary_IP
from pymor.parameters.base import ParameterSpace

# general options           
N = 10                                                                        # FE Dofs = (N+1)^2                                                
tol = 1e-14                                                                    # safeguard tolerance
tau = 3.5                                                                      # discrepancy parameter
k_max = 50                                                                     # maxit
noise_level = 1e-5        

# choose optimization methods
opt_methods = {     
              "Qr IRGNM",
              "FOM IRGNM",
              "Qr-Vr TR IRGNM",
               }

# choose norm in the parameter space
norm_type = 'L2'

print('Construct problem..')                                                     
analytical_problem, q_exact, N, problem_type, exact_analytical_problem, energy_problem = whole_problem(
                                                        N = N,
                                                        parameter_location = 'reaction',
                                                        boundary_conditions = 'dirichlet',
                                                        exact_parameter = 'Kirchner',
                                                       )



print('Discretizing problem...')                                                
diameter = np.sqrt(2)/N
grid_type = RectGrid
low = 0.001                                                                     # lower bound value
up = 1e20                                                                       # upper bound value 
par_dim = (N+1)**2                                                              # dimension of parameter space
q_low = low*np.ones((par_dim,))                                                 # lower bound
q_up =  up*np.ones((par_dim,))                                                  # upper bound
bounds = [q_low, q_up]
q0 = 3*np.ones((par_dim,))                                                      # initial guess for IRGNM
q_0 =  3*np.ones((par_dim,))                                                    # regularization center for IRGNM
opt_data = {'FE_dim': (N+1)**2,                                                 # FE dofs
            'par_dim': par_dim,                                                     
            'noise_level': noise_level,
            'q_exact': q_exact,
            'q0': q0,
            'q_0': q_0, 
            'low': low,
            'up': up,
            'noise_percentage': None,   
            'problem_type': problem_type,
            'q_in_h1': True,
            'norm_on_q': norm_type,
            'B_assemble': False,                                                # options to preassemble affine components or not
            'bounds': bounds
            } 

# create parameter space
parameter_space = ParameterSpace(analytical_problem.parameters, [low,up]) 

# discretize analytical problem to obtain inverse problem fom
fom_IP, fom_IP_data = discretize_instationary_IP(analytical_problem,
                                               diameter = diameter,
                                               opt_data = opt_data,
                                               exact_analytical_problem = exact_analytical_problem,
                                               energy_product_problem = energy_problem,
                                               grid_type = grid_type
                                               )

#print(analytical_problem.)
