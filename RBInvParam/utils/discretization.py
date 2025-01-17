from typing import Dict,Tuple
import numpy as np
np.random.seed(0)

from pymor.models.basic import InstationaryModel
from pymor.vectorarrays.interface import VectorArray
from pymor.operators.constructions import LincombOperator
from pymor.operators.interface import Operator
from pymor.parameters.functionals import ProjectionParameterFunctional, ParameterFunctional
from pymor.operators.numpy import NumpyMatrixOperator
from scipy.sparse import csr_matrix

from RBInvParam.products import BochnerProductOperator

def construct_noise_data(model : InstationaryModel,
                         q_exact : np.ndarray,
                         noise_level : float,
                         product: Operator, 
                         time_depend_noise: bool = True) -> Tuple[VectorArray, float]:

    u_exact = model.solve_state(q_exact)
    if time_depend_noise:
        noise = np.random.rand(len(u_exact), u_exact.dim)
        assert isinstance(product, BochnerProductOperator)
    else:
        noise = np.random.rand(1, u_exact.dim)
    
    noise = model.V.make_array(noise)
    noise_norm = np.sqrt(product.apply2(noise,noise))[0,0]
    assert noise_norm > 0
    
    noise_scaling = noise_level/noise_norm * noise
    u_noise = u_exact + noise_scaling    
    percentage = noise_level/noise_norm

    u_noise = u_exact + noise_scaling
    return u_noise, percentage


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
