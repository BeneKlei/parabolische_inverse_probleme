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
from RBInvParam.utils.discretization import split_constant_and_parameterized_operator, construct_noise_data
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.products import BochnerProductOperator
from RBInvParam.model import InstationaryModelIP
from RBInvParam.error_estimator import CoercivityConstantEstimator

def discretize_instationary_IP(analytical_problem : InstationaryProblem, 
                               setup : Dict,
                               logger: logging.Logger = None) -> Dict:

    if logger:
        logger = logger
    else:
        logger = get_default_logger(inspect.getframeinfo(inspect.currentframe()).function)
        logger.setLevel(logging.DEBUG)              
                            
    ############################### FOM ###############################
    primal_fom, primal_fom_data = discretize_instationary_cg(analytical_problem,
                                                             diameter=setup['dims']['diameter'],
                                                             preassemble= False,
                                                             grid_type = RectGrid,
                                                             nt = setup['dims']['nt'])
                                                             
    Q_h = NumpyVectorSpace(dim = setup['dims']['par_dim'], id='PARAM')
    V_h = primal_fom.operator.source
    
    assert Q_h.dim == primal_fom.operator.source.dim
    assert Q_h.dim == primal_fom.operator.range.dim

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

    assembled_products = {
        'l2' : primal_fom.products['l2'].assemble().matrix, 
        'h1' : primal_fom.products['h1'].assemble().matrix
    }

    # process products dict
    product_names = {}
    for (key,value) in setup['model_parameter']['products'].items():
        buf = value.split('_')
        assert len(buf) <= 2
        if len(buf) == 2:
            assert buf[0] == 'bochner'
        assert buf[-1] in assembled_products.keys()
        product_names[key] = buf[-1]

    products['prod_H'] = NumpyMatrixOperator(
        matrix = assembled_products[product_names['prod_H']],
        source_id = V_h.id,
        range_id = V_h.id
    )

    products['prod_Q'] = NumpyMatrixOperator(
        matrix = assembled_products[product_names['prod_Q']],
        source_id = Q_h.id,
        range_id = Q_h.id
    )
    products['prod_V'] = NumpyMatrixOperator(
        matrix = assembled_products[product_names['prod_V']],
        source_id = V_h.id,
        range_id = V_h.id
    )

    products['prod_C'] = NumpyMatrixOperator(
        matrix = assembled_products[product_names['prod_C']],
        source_id = V_h.id,
        range_id = V_h.id
    )


    products['bochner_prod_Q'] = BochnerProductOperator(
        product=NumpyMatrixOperator(
            matrix = assembled_products[product_names['bochner_prod_Q']],
            source_id = Q_h.id,
            range_id = Q_h.id
        ),
        delta_t=setup['model_parameter']['delta_t'],
        space = Q_h,
        nt = setup['dims']['nt']
    )

    products['bochner_prod_V'] = BochnerProductOperator(
        product=NumpyMatrixOperator(
            matrix = assembled_products[product_names['bochner_prod_V']],
            source_id = V_h.id,
            range_id = V_h.id
        ),
        delta_t=setup['model_parameter']['delta_t'],
        space = V_h,
        nt = setup['dims']['nt']
    )
    ############################### Operators ###############################

    u_0 = primal_fom.initial_data.as_range_array()
    M = primal_fom.mass

    t = setup['model_parameter']['T_initial']
    # The rhs is assumed to NOT depend on q
    assert len(primal_fom.rhs.parameters) in [0,1]
    if 't' not in primal_fom.rhs.parameters:
        L = primal_fom.rhs.as_range_array()
    else:
        L = V_h.zeros(reserve=setup['dims']['nt'])
        mu = Mu()
        for n in range(setup['dims']['nt']): 
            t += setup['model_parameter']['delta_t']
            mu = mu.with_(t=t)
            L[n] = primal_fom.rhs.as_range_array(mu)
    
    _, constant_operator = split_constant_and_parameterized_operator(
        primal_fom.operator
    )


    A = UnAssembledA(
        constant_operator = constant_operator,
        reaction_problem = ('reaction' in setup['model_parameter']['problem_type']),
        grid = primal_fom_data['grid'],
        boundary_info = primal_fom_data['boundary_info'],
        source=V_h,
        range=V_h,
        Q = Q_h
    )
    
    B = UnAssembledB(
        reaction_problem = ('reaction' in setup['model_parameter']['problem_type']),
        grid = primal_fom_data['grid'],
        boundary_info = primal_fom_data['boundary_info'],
        source=Q_h,
        range=V_h,
        V = V_h
    )

    ############################### Coercivity ###############################

            
    problem_type = setup['model_parameter']['problem_type']
    assert product_names['prod_V'] == 'h1'
    
    if 'dirichlet' in problem_type and 'diffusion' in problem_type:
        A_coercivity_constant_estimator_function = lambda q: abs(min(q.to_numpy()[0]))
    elif 'dirichlet' in problem_type and 'reaction' in problem_type:
        A_coercivity_constant_estimator_function = lambda q: 1
    else:
        raise ValueError('No matching problemtype given')

    A_coercivity_constant_estimator = CoercivityConstantEstimator(
        coercivity_estimator_function = A_coercivity_constant_estimator_function,
        Q = Q_h,
        q_time_dep = setup['model_parameter']['q_time_dep']
    )
    
    ############################### Regularization ###############################

    q_circ = setup['model_parameter']['q_circ']
    assert type(q_circ) == np.ndarray
    q_circ = Q_h.make_array(q_circ)
    assert len(q_circ) in [setup['dims']['nt'], 1]

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

    ############################### Dummy Model ###############################

    setup['model_parameter']['q_exact'] = Q_h.make_array(setup['model_parameter']['q_exact'])

    building_blocks = {
        'u_0' : u_0, 
        'M' : M,
        'A' : A,
        'L' : L,
        'B' : B,
        'constant_cost_term' : None,
        'linear_cost_term' : None,
        'bilinear_cost_term' : None,
        'Q' : Q_h,
        'V' : V_h,
        'q_circ' : q_circ,
        'constant_reg_term' : constant_reg_term,
        'linear_reg_term' : linear_reg_term,
        'bilinear_reg_term' : bilinear_reg_term,
        'state_error_estimator' : None,
        'adjoint_error_estimator' : None,
        'objective_error_estimator' : None,
        'products' : products,
        'visualizer' : visualizer,
        'model_constants' : None,
        'setup' : setup
    }

    dummy_model = InstationaryModelIP(                 
        **building_blocks,
    )

    ############################### Cost ###############################

    C = NumpyMatrixOperator(
        scipy.sparse.identity(V_h.dim),
        source_id = V_h.id,
        range_id = V_h.id,
    )
    C_continuity_constant = 1.0

        
    u_delta, percentage = construct_noise_data(model = dummy_model, 
                                               q_exact = setup['model_parameter']['q_exact'],
                                               noise_level = setup['model_parameter']['noise_level'],
                                               product=products['bochner_prod_V'],
                                               time_depend_noise=True)
    y_delta = C.apply(u_delta)
    assert (len(y_delta) == setup['dims']['nt'])
    assert (y_delta.space == C.range) 

    logger.debug(f'noise percentage is {percentage:3.4e}')
    logger.debug(f'noise_level is {setup["model_parameter"]["noise_level"]:3.4e}')

    constant_cost_term = y_delta.pairwise_inner(y_delta, product=products['prod_C'])
    linear_cost_term = NumpyMatrixOperator(
        matrix = C.matrix.T @ products['prod_C'].assemble().matrix @ y_delta.to_numpy().T,
        source_id = None,
        range_id = V_h.id
    )
    bilinear_cost_term = NumpyMatrixOperator(
        matrix = C.matrix.T @ products['prod_C'].assemble().matrix @ C.matrix,
        source_id = V_h.id,
        range_id = V_h.id
    )

    ############################### Final ###############################

    building_blocks['constant_cost_term'] = constant_cost_term
    building_blocks['linear_cost_term'] = linear_cost_term
    building_blocks['bilinear_cost_term'] = bilinear_cost_term
    building_blocks['model_constants'] = {
        'A_coercivity_constant_estimator' : A_coercivity_constant_estimator,
        'C_continuity_constant' : C_continuity_constant
    }

    return building_blocks