import numpy as np
import scipy
from typing import Dict
import logging
import inspect

import pymor.models.basic as InstationaryProblem
from pymor.discretizers.builtin import discretize_instationary_cg
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.discretizers.builtin.grids.rect import RectGrid
from pymor.parameters.base import Mu

from RBInvParam.evaluators import UnAssembledA, UnAssembledB
from RBInvParam.utils.discretization import split_constant_and_parameterized_operator, \
    construct_noise_data
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.products import BochnerProductOperator

# TODO Refactor this
# def bochner_product(v : VectorArray,
#                     w : VectorArray,
#                     delta_t : float,
#                     product : NumpyMatrixOperator) -> float:

#     assert product.source == product.range
#     assert v in product.range
#     assert w in product.range
#     assert len(v) == len(w)

#     return np.sum(delta_t * product.pairwise_apply2(v,w))


def discretize_instationary_IP(analytical_problem : InstationaryProblem, 
                               model_params : Dict,
                               dims : Dict,
                               problem_type: str,
                               logger: logging.Logger = None) -> Dict:

    if logger:
        logger = logger
    else:
        logger = get_default_logger(inspect.getframeinfo(inspect.currentframe()).function)
        logger.setLevel(logging.DEBUG)              
                            
    ############################### FOM ###############################
    primal_fom, primal_fom_data = discretize_instationary_cg(analytical_problem,
                                               diameter=dims['diameter'],
                                               preassemble= False,
                                               grid_type = RectGrid,
                                               nt = dims['nt'])
    # helper
    source = primal_fom.operator.source
    range = primal_fom.operator.range


    Q_h = NumpyVectorSpace(dim = dims['par_dim'], id='PARAM')
    V_h = primal_fom.operator.source

    assert Q_h.dim == source.dim
    assert Q_h.dim == range.dim

    visualizer = primal_fom.visualizer
    
    ############################### Products ###############################

    products = {
        'prod_H' : None,
        'prod_Q' : None,
        'prod_V' : None,
        'prod_C' : None,
        'bochner_prod_Q' : None,
        'bochner_prod_V' : None,
    }


    # TODO
    raise NotImplementedError    
    for prod_name in model_params['products'].values():
        cond_1 = prod_name in ['h1', 'l2']
        
    

    l2 = primal_fom.products['l2'].assemble().matrix
    h1 = primal_fom.products['h1'].assemble().matrix

    products['prod_H'] = primal_fom.products[model_params['products']['prod_H']]
    products['prod_Q'] = NumpyMatrixOperator(
        matrix = primal_fom.products['l2'].assemble().matrix,
        source_id = Q_h.id,
        range_id = Q_h.id
    )
    products['prod_V'] = primal_fom.products['h1']
    products['prod_C'] = primal_fom.products['l2'].assemble().matrix
    

    delta_t = model_params['delta_t']
    products['bochner_prod_Q'] = BochnerProductOperator(
        product=products['prod_Q'],
        delta_t=delta_t
    )
    products['bochner_prod_V'] = BochnerProductOperator(
        product=products['prod_V'],
        delta_t=delta_t
    )

    ############################### Operators ###############################

    u_0 = primal_fom.initial_data.as_range_array()
    M = primal_fom.mass

    t = model_params['T_initial']
    # The rhs is assumed to NOT depend on q
    assert len(primal_fom.rhs.parameters) in [0,1]
    if 't' not in primal_fom.rhs.parameters:
        L = primal_fom.rhs.as_range_array()
    else:
        L = V_h.zeros(reserve=dims['nt'])
        mu = Mu()
        for n in range(dims['nt']): 
            t += model_params['delta_t']
            mu = mu.with_(t=t)
            L[n] = primal_fom.rhs.as_range_array(mu)

    _, constant_operator = split_constant_and_parameterized_operator(
        primal_fom.operator
    )

    A = UnAssembledA(
        constant_operator = constant_operator,
        reaction_problem = ('reaction' in problem_type),
        grid = primal_fom_data['grid'],
        boundary_info = primal_fom_data['boundary_info'],
        source=V_h,
        range=V_h,
        Q = Q_h,
        dims = dims
    )
    
    B = UnAssembledB(
        reaction_problem = ('reaction' in problem_type),
        grid = primal_fom_data['grid'],
        boundary_info = primal_fom_data['boundary_info'],
        source=Q_h,
        range=V_h,
        V = V_h,
        dims = dims
    )

    ############################### Cost ###############################

    C = NumpyMatrixOperator(
        scipy.sparse.identity(range.dim),
        source_id = source.id,
        range_id = range.id,
    )
        
    u_delta, percentage = construct_noise_data(primal_fom, model_params)
    y_delta = C.apply(u_delta)[1:]
    assert (len(y_delta) == dims['nt'])
    assert (y_delta.space == C.range) 

    logger.debug(f'noise percentage is {percentage:3.4e}')
    logger.debug(f'noise_level is {model_params["noise_level"]:3.4e}')

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
        source_id = Q_h.id,
        range_id = Q_h.id
    )
    bilinear_reg_term = NumpyMatrixOperator(
        matrix = products['prod_Q'].matrix,
        source_id = Q_h.id,
        range_id = Q_h.id
    )

    building_blocks = {
        'u_0' : u_0, 
        'M' : M,
        'A' : A,
        'L' : L,
        'B' : B,
        'constant_cost_term' : constant_cost_term,
        'linear_cost_term' : linear_cost_term,
        'bilinear_cost_term' : bilinear_cost_term,
        'Q' : Q_h,
        'V' : V_h,
        'q_circ' : q_circ,
        'constant_reg_term' : constant_reg_term,
        'linear_reg_term' : linear_reg_term,
        'bilinear_reg_term' : bilinear_reg_term,
        'state_error_estimator' : None,
        'adjoint_error_estimator' : None,
        'products' : products,
        'visualizer' : visualizer
    }

    assert all(v is not None for v in building_blocks) 
    return building_blocks

    





 
