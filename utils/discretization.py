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

    q_exact = np.array([model_params['q_exact'][0]])    
    assert len(q_exact) == 1
        
    u_exact = analytical_problem.solve(q_exact)    
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


def split_constant_and_parameterized_operator(
        complete_operator : LincombOperator
    ):
    assert isinstance(complete_operator, LincombOperator)
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
    if isinstance(matrix, csr_matrix):
        matrix.eliminate_zeros()
        
    constant_operator = NumpyMatrixOperator(
        matrix = matrix,
        source_id = complete_operator.source.id,
        range_id = complete_operator.range.id
    )
    
    return parameterized_operator, constant_operator

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
