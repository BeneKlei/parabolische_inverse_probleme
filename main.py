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
from gradient_descent import gradient_descent_non_linearized_problem

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)

set_log_levels({
    'pymor' : 'WARN'
})

set_defaults({
    # 'pymor.algorithms.gram_schmidt.gram_schmidt.atol' : 1e-15,
    # 'pymor.algorithms.gram_schmidt.gram_schmidt.rtol' : 1e-15,
})

N = 10                                                                      # FE Dofs = (N+1)^2                                                
noise_level = 0
nt = 50
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
    'noise_level' : noise_level,
    'q_circ' : q_circ, 
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
                                             problem_type) 

FOM = InstationaryModelIP(                 
    *building_blocks,
    dims = dims,
    model_parameter = model_parameter
)

if 0:
    # Gradient tests 
    
    #objective 
    FOM.derivative_check(FOM.compute_objective, FOM.compute_gradient)
    
    # gradient regularization term
    alpha = 1e0
    FOM.derivative_check(lambda q: alpha * FOM.regularization_term(q), lambda q: alpha * FOM.gradient_regularization_term(q))
    
    # linarized objective
    alpha = 1e-0
    q = FOM.numpy_to_pymor(q_circ)
    FOM.derivative_check(lambda d : FOM.compute_linearized_objective(q, d, alpha), lambda d: FOM.compute_linearized_gradient(q, d, alpha))

    # # Gradient descent test
    print("Solve optimization problem")

    q_est = gradient_descent_non_linearized_problem(FOM, q_start=q_circ, alpha=0, max_iter=1e5, tol=1e-12, inital_step_size=1e8, logger=logger)

    q_exact = []
    for idx in range(dims['nt']):
        q_exact.append(model_parameter['q_exact'])
    
    q_exact = FOM.Q.make_array(np.array(q_exact))
    print("Differnce to q_exact:")
    print("L^inf") 
    print(np.max(np.abs((q_est - q_exact).to_numpy())))
    print("Q-Norm") 
    print(np.sqrt(np.sum(FOM.products['prod_Q'].pairwise_apply2(q_est - q_exact, q_est - q_exact))))

    # d_start = q_circ.copy()
    # d_start[:,:] = 0
    # d_start = FOM.Q.make_array(d_start)

    # d_est = gradient_descent_linearized_problem(FOM, q=FOM.Q.make_array(q_circ), d_start=d_start, alpha=0, max_iter=1e5, tol=1e-12, inital_step_size=1e8)
        
if 1:
    q_start = 0*np.ones((nt, dims['par_dim']))
    optimizer_parameter = {
        'noise_level' : model_parameter['noise_level'],
        'tau' : 1,
        'tol' : 1e-9,
        'q_0' : q_start,
        'alpha_0' : 1e-3,
        'i_max' : 50,
        'reg_loop_max' : 50,
        'theta' : 0.25,
        'Theta' : 0.75
    }

    optimizer = FOMOptimizer(
        FOM = FOM,
        optimizer_parameter = optimizer_parameter
    )
    optimizer.logger.setLevel(logging.DEBUG)
    q_est = optimizer.solve()

    q_exact = FOM.Q.make_array([np.array(q_exact) for _ in range(dims['nt'])])
    FOM.visualizer.visualize(q_est, title="q_est")
    FOM.visualizer.visualize(q_exact, title="q_exact")
    print("Differnce to q_exact:")
    print("L^inf") 
    print(np.max(np.abs((q_est - q_exact).to_numpy())))
    print("Q-Norm") 
    norm_delta_q = np.sqrt(np.sum(FOM.products['prod_Q'].pairwise_apply2(q_est - q_exact, q_est - q_exact)))
    norm_q_exact = np.sqrt(np.sum(FOM.products['prod_Q'].pairwise_apply2(q_exact, q_exact)))
    print(norm_delta_q)
    print(norm_q_exact)