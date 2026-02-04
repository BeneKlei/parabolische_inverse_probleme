from typing import Dict,Tuple
import numpy as np
np.random.seed(0)

from pymor.vectorarrays.interface import VectorArray
from pymor.operators.constructions import LincombOperator
from pymor.operators.interface import Operator
from pymor.parameters.functionals import ProjectionParameterFunctional, ParameterFunctional
from pymor.operators.numpy import NumpyMatrixOperator
from scipy.sparse import csr_matrix

#from RBInvParam.model import InstationaryModelIP

def construct_noise_data(model,
                         q_exact : np.ndarray,
                         C: Operator,
                         noise_level : float,
                         product: Operator, 
                         time_depend_noise: bool = True) -> Tuple[VectorArray, VectorArray]:


    u_exact = model.solve_state(q_exact)
    y_exact = C.apply(u_exact)

    if time_depend_noise:
        noise = C.range.random(len(y_exact))
    else:
        noise = C.range.random(1)

    #noise_norm = np.sqrt(product.apply2(y_exact,y_exact))[0,0]
    noise_norm = np.sqrt(product.apply2(noise,noise))[0,0]
    assert noise_norm > 0
    
    noise_scaling = noise_level/noise_norm * noise
    y_noise = y_exact + noise_scaling    

    return y_noise, u_exact

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

# TODO Maybe refactor this mechanic
def split_constant_and_parameterized_operator(
        complete_operator : LincombOperator
    ):
    print(complete_operator)
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

    if len(constant_operators) == 0:
        assert len(constant_coefficients) == 0
        constant_operator = None
    else:
        constant_operator = LincombOperator(constant_operators, constant_coefficients).assemble()
        matrix = constant_operator.matrix.copy()
        if isinstance(matrix, csr_matrix):
            matrix.eliminate_zeros()
            
        constant_operator = NumpyMatrixOperator(
            matrix = matrix
        )

    parameterized_operator = LincombOperator(operators, coefficients, name='true_parameterized_operator')

    
    
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


def process_product_names(products: Dict) -> Dict:
    product_names = {}
    product_name = ''
    for (key,value) in products.items():
        buf = value.split('_')
        if buf[0] == 'bochner':
            product_name = '_'.join(buf[1:])
        else:
            product_name = value
        product_names[key] = product_name
    return product_names
