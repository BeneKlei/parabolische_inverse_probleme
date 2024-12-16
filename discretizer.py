import numpy as np
import scipy
from typing import Dict

import pymor.models.basic as InstationaryProblem
from pymor.discretizers.builtin import discretize_instationary_cg
from pymor.operators.constructions import IdentityOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
from utils import construct_noise_data
from pymor.discretizers.builtin.grids.rect import RectGrid
from evaluators import UnAssembledA, UnAssembledB

from utils import split_constant_and_parameterized_operator

def discretize_instationary_IP(analytical_problem : InstationaryProblem, 
                               model_params : Dict,
                               dims : Dict,
                               problem_type: str
                               ):
    products = {
        'prod_H' : None,
        'prod_Q' : None,
        'prod_V' : None,
        'prod_C' : None
    }

    ############################### PDE ###############################
    primal_fom, primal_fom_data = discretize_instationary_cg(analytical_problem,
                                               diameter=dims['diameter'],
                                               preassemble= False,
                                               grid_type = RectGrid,
                                               nt = dims['nt'])
    # helper
    source = primal_fom.operator.source
    range = primal_fom.operator.range


    Q_h = NumpyVectorSpace(dim = dims['par_dim'], id='PARAM')
    assert Q_h.dim == source.dim
    assert Q_h.dim == range.dim

    products['prod_Q'] = NumpyMatrixOperator(
        matrix = primal_fom.products['l2'].assemble().matrix,
        source_id = Q_h.id,
        range_id = Q_h.id
    )

    V_h = primal_fom.operator.source
    products['prod_V'] = primal_fom.products['h1']
    products['prod_H'] = primal_fom.products['l2']


    visualizer = primal_fom.visualizer
    
    u_0 = primal_fom.initial_data.as_range_array()
    M = primal_fom.mass
    f = primal_fom.rhs

    _, constant_operator = split_constant_and_parameterized_operator(
        primal_fom.operator
    )

    A = UnAssembledA(
        constant_operator = constant_operator,
        reaction_problem = ('reaction' in problem_type),
        grid = primal_fom_data['grid'],
        boundary_info = primal_fom_data['boundary_info'],
        Q = Q_h,
        dims = dims
    )
    
    B = UnAssembledB(
        reaction_problem = ('reaction' in problem_type),
        grid = primal_fom_data['grid'],
        boundary_info = primal_fom_data['boundary_info'],
        V = V_h,
        dims = dims
    )

    ############################### Cost ###############################

    if 1:
        products['prod_C'] = primal_fom.products['l2']
        C = NumpyMatrixOperator(
            scipy.sparse.identity(range.dim),
            source_id = source.id,
            range_id = range.id,
        )
        
    u_delta = construct_noise_data(primal_fom, model_params)
    y_delta = C.apply(u_delta)[1:]
    assert (len(y_delta) == dims['nt'])
    assert (y_delta.space == C.range) 

    constant_cost_term = y_delta.pairwise_inner(y_delta, product=products['prod_C'])    
    linear_cost_term = NumpyMatrixOperator(
        matrix = C.matrix.T @ products['prod_C'].assemble().matrix @ y_delta.to_numpy().T,
        source_id = None,
        range_id = range.id
    )
    bilinear_cost_term = NumpyMatrixOperator(
        matrix = C.matrix.T @ products['prod_C'].assemble().matrix @ C.matrix,
        source_id = source.id,
        range_id = range.id
    )

    ############################### Regularization ###############################

    q_circ = model_params['q_circ']
    assert type(q_circ) == np.ndarray
    q_circ = Q_h.make_array(q_circ)
    assert len(q_circ) in [dims['nt'], 1]

    constant_reg_term = q_circ.pairwise_inner(q_circ, product=products['prod_Q'])    
    linear_reg_term = NumpyMatrixOperator(
        matrix = products['prod_Q'].matrix.T @ q_circ.to_numpy().T,
        source_id =  Q_h.id,
        range_id = Q_h.id
    )
    bilinear_reg_term = NumpyMatrixOperator(
        matrix = products['prod_Q'].matrix,
        source_id =  Q_h.id,
        range_id = Q_h.id
    )
    
    building_blocks = (
        u_0, 
        M,
        A,
        f,
        B,
        constant_cost_term,
        linear_cost_term,
        bilinear_cost_term,
        Q_h,
        V_h,
        q_circ,
        constant_reg_term,
        linear_reg_term,
        bilinear_reg_term,
        products,
        visualizer
    )

    assert all(v is not None for v in building_blocks) 
    return building_blocks

    





 
