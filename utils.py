from typing import Dict
import numpy as np
np.random.seed(0)

import pymor.models.basic as InstationaryModel
from pymor.operators.constructions import LincombOperator
from pymor.parameters.functionals import ProjectionParameterFunctional, \
    ParameterFunctional
from pymor.operators.numpy import NumpyMatrixOperator
from scipy.sparse import csr_matrix

def construct_noise_data(analytical_problem : InstationaryModel, 
                         model_params : Dict):

    assert 'q_exact' in model_params
    u_exact = analytical_problem.solve(model_params['q_exact'])    
    noise = np.random.rand(u_exact.dim,)   
    noise_np = u_exact.space.from_numpy(noise)
    noise_L2_norm = np.sqrt(analytical_problem.products['l2'].apply2(noise_np,noise_np))[0,0]  # get l2 norm of noise
    
    if 1: # insert noise level                                                                
        noise_level = model_params["noise_level"]
        noise_scaling = u_exact.space.from_numpy(noise_level*noise/noise_L2_norm) 
        u_noise = u_exact + noise_scaling    
        percentage = noise_level/noise_L2_norm
        print(f'noise percentage is {percentage}')
        print(f'noise_level is {noise_level}')


    u_noise = u_exact + noise_scaling
    return u_noise

def build_projection(grid):
    rows = []
    cols = []
    data = []
    cols_switched = []
    nodes_per_axis_t = int(np.sqrt(len(grid.centers(0)))) # N
    nodes_per_axis_n = int(np.sqrt(len(grid.centers(2)))) # N+1
    for i in range(len(grid.centers(0))):
        j = i // nodes_per_axis_t
        entries = [i + j, i + j + 1, i + j + nodes_per_axis_n, i + j + nodes_per_axis_n + 1]
        rows.extend([i, i, i, i])
        cols.extend(entries)
        data.extend([1 / 4., 1 / 4., 1 / 4., 1 / 4.])
        # cols switched in order of shape functions (lower left, lower right, upper right, upper left)
        entries_switched = [entries[0],entries[1], entries[3], entries[2]]
        cols_switched.extend(entries_switched)  
    nodes_to_element_projection = csr_matrix((data, (rows, cols)))
    return nodes_to_element_projection, cols, cols_switched

def split_constant_and_parameterized_operator(complete_operator):
    operators, coefficients = [], []
    constant_operators, constant_coefficients = [], []
    for coef, op in zip(complete_operator.coefficients, complete_operator.operators):
        assert not op.parametric, 'B operator needs to be a true LincombOperator'
        if isinstance(coef, ParameterFunctional) and coef.parametric:
            # then the operator is parametric
            assert isinstance(coef, ProjectionParameterFunctional), 'other cases are not implemented yet'
            operators.append(op)
            coefficients.append(coef)
        else:
            constant_operators.append(op)
            constant_coefficients.append(coef)
    constant_operator = LincombOperator(constant_operators, constant_coefficients).assemble()
    parameterized_operator = LincombOperator(operators, coefficients, name='true_parameterized_operator')

    matrix = constant_operator.matrix.copy()
    matrix.eliminate_zeros()
    constant_operator = NumpyMatrixOperator(
        matrix = matrix,
        source_id = complete_operator.source.id,
        range_id = complete_operator.range.id
    )

    return parameterized_operator, constant_operator

    # N = int(1/diameter*np.sqrt(2)) 
    # refinement_factor = 2
    # N_fine = N*refinement_factor
    # diameter_fine = np.sqrt(2)/N_fine
    # exact_assembled_fom, exact_assembled_fom_data = discretize_stationary_cg(exact_analytical_problem,
    #                                                        diameter= diameter_fine,#int(diameter/refinement_factor),
    #                                                        grid_type=grid_type,
    #                                                        preassemble= True
    #                                                        )
    
    # # assemble energy product coreesponding to parameter q = 1
    # q_np = np.ones((opt_data['par_dim'],))
    # q_energy_product = analytical_problem.parameters.parse(q_np)
    # energy_product_fom, _ = discretize_stationary_cg(energy_product_problem,
    #                                                        diameter=diameter,
    #                                                        grid_type=grid_type,
    #                                                        preassemble= True,
    #                                                        mu_energy_product = q_energy_product
    #                                                        )
    # products = energy_product_fom.products.copy()
    
    # # construct exact and noisy data
    # u_exact_fine = exact_assembled_fom.solve()                                  # get exact data
    # interpolate_inds = interpolate_between_grids(N_fine, refinement_factor)     
    # u_exact = u_exact_fine.to_numpy()[0][interpolate_inds]                      # interpolate it on the grid             
    # u_exact = assembled_model.solution_space.from_numpy(u_exact)
    np.random.seed(0)                                                           # fix random seed
    noise = np.random.rand(u_exact.dim,)                                        # get noise
    noise_L2_norm = np.sqrt(assembled_model.products['l2'].apply2(
        u_exact.space.from_numpy(noise),u_exact.space.from_numpy(noise)))[0,0]  # get l2 norm of noise
    
    if 1: # insert noise level                                                                
        noise_level = opt_data["noise_level"]
        noise_scaling = u_exact.space.from_numpy(noise_level*noise/noise_L2_norm) 
        u_noise = u_exact + noise_scaling                                       # get noisy data 
        # percentage = noise_level/noise_L2_norm
        # opt_data["percentage"] = percentage
        # print(f'noise percentage is {percentage}')
        # print(f'noise_level is {noise_level}')
    
    # else: # insert percentage
    #     assert  opt_data["noise_percentage"] is not None, 'please give a valid noise percentage...'
    #     percentage = opt_data["noise_percentage"]
    #     u_exact_l2_norm = assembled_model.l2_norm(u_exact)[0]
    #     noise_level = u_exact_l2_norm*percentage
    #     opt_data["noise_level"] = noise_level
    #     # print(f'noise percentage is {percentage}')
    #     # print(f'noise_level is {noise_level}')
    #     noise_scaling = u_exact.space.from_numpy(noise_level*noise/noise_L2_norm)
    #     u_noise = u_exact + noise_scaling  
   
    # # clear dirichlet dofs from u_noise
    # if 'diricsuper.__init__(*locals())primal_fom_data["boundary_info"].dirichlet_boundaries(2)
    #     u_noise[DI] = 0
    #     u_noise = u_exact.space.from_numpy(u_noise)



def interpolate_between_grids(N_fine, refinement_factor):
    right = [i*(N_fine+1) for i in range(1,N_fine+2)]
    left = [i-N_fine for i in right]
    left = left[::refinement_factor]
    right = right[::refinement_factor]
    indices_coarse = []
    for i in range(len(left)):                                                 
        inds_x_axis = list(range(left[i], right[i]+1))
        indices_coarse.extend(inds_x_axis[::refinement_factor])  
    N = N_fine/refinement_factor
    assert len(indices_coarse) == (N+1)**2, 'wrong dimensions...'
    return [i-1 for i in indices_coarse]

class Struct():
    pass
