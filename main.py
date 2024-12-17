import numpy as np
import logging
from pathlib import Path
import sys
sys.path.append('../')

from pymor.basic import *
from pymor.parameters.base import ParameterSpace

from model import InstationaryModelIP
from optimizer import FOMOptimizer
from problems import whole_problem
from discretizer import discretize_instationary_IP
from helpers import save_dict_to_pkl

#from gradient_descent import gradient_descent_non_linearized_problem

# TODO
# - Find better way to handle time independ parameter

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
par_dim = (N+1)**2
fine_N = 2 * N

T_initial = 0
T_final = 1
nt = 50
delta_t = (T_final - T_initial) / nt
q_time_dep = True

noise_level = 0

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


print('Construct problem..')                                                     
analytical_problem, q_exact, N, problem_type, exact_analytical_problem, energy_problem = whole_problem(
                                                        N = N,
                                                        parameter_location = 'reaction',
                                                        boundary_conditions = 'dirichlet',
                                                        exact_parameter = 'Kirchner',
                                                       )

model_parameter['parameters'] = analytical_problem.parameters
if q_time_dep:                                                 
    model_parameter['q_exact'] = np.array([q_exact for _ in range(dims['nt'])])
else:
    model_parameter['q_exact'] = q_exact

print('Discretizing problem...')                                                
building_blocks = discretize_instationary_IP(analytical_problem,
                                             model_parameter,
                                             dims, 
                                             problem_type) 



FOM = InstationaryModelIP(                 
    *building_blocks,
    dims = dims,
    model_parameter = model_parameter
)


# q_exact = FOM.Q.make_array(model_parameter['q_exact'])
# print(FOM.compute_objective(q_exact))
# q_delta = FOM.Q.make_array(np.random.random((nt, dims['par_dim'])))
# Eps = np.array([1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6])
# for e in Eps:
#     print("---------------------------------------")
#     print(e)
#     print(FOM.compute_objective(q_exact + e * q_delta))
#     print(np.sqrt(FOM.products['bochner_prod_Q'](e * q_delta, e * q_delta)))
#     print(np.sqrt(FOM.products['bochner_prod_Q'](e * q_delta, e * q_delta)) / np.sqrt(FOM.products['bochner_prod_Q'](q_exact, q_exact)) * 100)

# print("###############################################")
# q_exact = FOM.Q.make_array(model_parameter['q_exact'])
# print(FOM.compute_objective(q_exact))
# q_delta = q_exact
# Eps = np.array([1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6])
# for e in Eps:
#     print("---------------------------------------")
#     print(e)
#     print(FOM.compute_objective(q_exact + e * q_delta))
#     print(np.sqrt(FOM.products['bochner_prod_Q'](e * q_delta, e * q_delta)))
#     print(np.sqrt(FOM.products['bochner_prod_Q'](e * q_delta, e * q_delta)) / np.sqrt(FOM.products['bochner_prod_Q'](q_exact, q_exact)) * 100)


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
        #'tol' : 1e-6,
        #'tol' : 1e-12,
        'tol' : 1e-14,
        'q_0' : q_start,
        'alpha_0' : 0,
        #'alpha_0' : 1e-3,
        'i_max' : 1,
        #'i_max' : 50,
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

    q_exact = FOM.Q.make_array(model_parameter['q_exact'])
    FOM.visualizer.visualize(q_est, title="q_est")
    FOM.visualizer.visualize(q_exact, title="q_exact")
    print("Differnce to q_exact:")
    print("L^inf") 
    print(f"{np.max(np.abs((q_est - q_exact).to_numpy())):3.4e}")
    print("Q-Norm") 
    norm_delta_q = np.sqrt(FOM.products['bochner_prod_Q'](q_est - q_exact, q_est - q_exact))
    norm_q_exact = np.sqrt(FOM.products['bochner_prod_Q'](q_exact, q_exact))
    print(f"Absolute error: {norm_delta_q:3.4e}")
    print(f"Relative error: {norm_delta_q / norm_q_exact * 100:3.4}%.")

    save_path = Path("./dumps/test.pkl")
    print(f"Save statistics to {save_path}")

    data = {
        'dims' : dims,
        'model_parameter' : model_parameter,
        'optimizer_parameter' : optimizer_parameter,
        'optimizer_statistics' : optimizer.statistics
    }

    save_dict_to_pkl(path=save_path, data=data)

    