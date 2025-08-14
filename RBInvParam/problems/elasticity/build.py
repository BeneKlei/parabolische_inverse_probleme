import numpy as np
import scipy
import logging
import inspect
from typing import Dict

from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
 
import pymor_dealii_bindings as pd2

import RBInvParam.problems.elasticity.material_model as mm

from RBInvParam.problems.elasticity.pymor_dealii_bindings.vectorarray import DealIIVectorSpace
from RBInvParam.problems.elasticity.pymor_dealii_bindings.operator import DealIIMatrixOperator
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.utils.discretization import construct_noise_data
from RBInvParam.model import InstationaryModelIP
from RBInvParam.products import BochnerProductOperator

from RBInvParam.problems.elasticity.evaluators import FOMEvaluatorA

def build_InstationaryModelIP(setup : Dict,
                              logger : logging.Logger = None) -> InstationaryModelIP:

    if logger:
        logger = logger
    else:
        logger = get_default_logger(inspect.getframeinfo(inspect.currentframe()).function)
        logger.setLevel(logging.DEBUG)   

    logger.debug('Construct problem..')

    material_model_config = mm.MaterialModelConfig()
    material_model_config.T_initial = setup['T_initial']
    material_model_config.T_final = setup['T_final']
    material_model_config.delta_t = setup['delta_t']
    material_model_config.nt = setup['nt']
    material_model_config.par_dim = setup['par_dim']

    material_model = mm.MaterialModel(material_model_config)
    material_model.make_grid()
    material_model.setup_system()

    setup['N'] = material_model.n_dofs()

    Q_h = NumpyVectorSpace(dim = setup['par_dim'])
    V_h = DealIIVectorSpace(dim = setup['N'])

    ############################### Products ###############################

    products = {
        'prod_H' : None,
        'prod_Q' : None,
        'prod_V' : None,
        'prod_C' : None,
        'bochner_prod_Q' : None,
        'bochner_prod_V' : None,
    }

    assembled_state_products = {
        'h1' : pd2.SparseMatrix(),
        'h1_semi' : pd2.SparseMatrix(),
        'l2' : pd2.SparseMatrix(),
        'h1_0' : pd2.SparseMatrix(),
        'h1_0_semi' : pd2.SparseMatrix(),
        'l2_0' : pd2.SparseMatrix(),
    }

    material_model.assemble_h1_matrix(assembled_state_products['h1'])
    material_model.assemble_h1_semi_matrix(assembled_state_products['h1_semi'])
    material_model.assemble_l2_matrix(assembled_state_products['l2'])
    material_model.assemble_h1_0_matrix(assembled_state_products['h1_0'])
    material_model.assemble_h1_0_semi_matrix(assembled_state_products['h1_0_semi'])
    material_model.assemble_l2_0_matrix(assembled_state_products['l2_0'])

    #process products dict
    product_names = {}
    product_name = ''
    for (key,value) in setup['products'].items():
        buf = value.split('_')
        if buf[0] == 'bochner':
            product_name = '_'.join(buf[1:])
        else:
            product_name = value
        product_names[key] = product_name

    assembled_parameter_products  = {
        'euclid' : scipy.sparse.identity(Q_h.dim)
    }

    # Assume H = V = C
    assert product_names['prod_H'] in assembled_state_products.keys()
    assert product_names['prod_Q'] in assembled_parameter_products.keys()
    assert product_names['prod_V'] in assembled_state_products.keys()
    assert product_names['prod_C'] in assembled_state_products.keys()
    assert product_names['bochner_prod_Q'] in assembled_parameter_products.keys()
    assert product_names['bochner_prod_V'] in assembled_state_products.keys()

    products['prod_H'] = DealIIMatrixOperator(
        matrix = assembled_state_products[product_names['prod_H']]
    )

    products['prod_Q'] = NumpyMatrixOperator(
        matrix = assembled_parameter_products[product_names['prod_Q']]
    )

    products['prod_V'] = DealIIMatrixOperator(
        matrix = assembled_state_products[product_names['prod_V']]
    )

    products['prod_C'] = DealIIMatrixOperator(
        matrix = assembled_state_products[product_names['prod_C']]
    )

    products['bochner_prod_Q'] = BochnerProductOperator(
        product=NumpyMatrixOperator(
            matrix = assembled_parameter_products[product_names['bochner_prod_Q']]
        ),
        delta_t=setup['delta_t'],
        space = Q_h,
        nt = setup['nt']
    )

    products['bochner_prod_V'] = BochnerProductOperator(
        product=DealIIMatrixOperator(
            matrix = assembled_state_products[product_names['bochner_prod_V']]
        ),
        delta_t=setup['delta_t'],
        space = V_h,
        nt = setup['nt']
    )

    ############################### Operators ###############################

    zero_data = V_h.zeros(1)
    initial_data = {
        'state' : {
            'zeroth_order' : zero_data,
            'first_order' : zero_data
        },
        'adjoint' : {
            'zeroth_order' : zero_data,
            'first_order' : zero_data
        },
        'lin_state' : {
            'zeroth_order' : zero_data,
            'first_order' : zero_data
        },
        'lin_adjoint' : {
            'zeroth_order' : zero_data,
            'first_order' : zero_data
        },
    }
    
    M = pd2.SparseMatrix()
    material_model.assemble_mass_matrix(M)
    M = DealIIMatrixOperator(
        matrix = M
    )

    L = V_h.make_array(material_model.get_force_list())

    A = FOMEvaluatorA(
        material_model = material_model,
        source = V_h,
        range = V_h,
        Q = Q_h,
    )

    B = None
    ############################### Coercivity ###############################

    A_coercivity_constant_estimator = None

    ############################### Regularization ###############################

    q_circ = setup['q_circ']
    assert type(q_circ) == np.ndarray
    q_circ = Q_h.make_array(q_circ)
    assert len(q_circ) in [setup['nt']+1, 1]

    constant_reg_term = q_circ.pairwise_inner(q_circ, product=products['prod_Q'])    
    linear_reg_term = NumpyMatrixOperator(
        matrix = products['prod_Q'].matrix.T @ q_circ.to_numpy().T
    )
    bilinear_reg_term = NumpyMatrixOperator(
        matrix = products['prod_Q'].matrix
    )

    ############################### u^delta / Dummy Model ###############################
    q_exact = setup['q_exact']
    assert type(q_exact) == np.ndarray
    q_exact = Q_h.make_array(q_exact)
    assert len(q_exact) in [setup['nt'] + 1, 1]

    building_blocks = {
        'initial_data' : initial_data, 
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
        'visualizer' : None,
        'model_constants' : None,
        'setup' : setup,
        'bounds' : setup['bounds']
    }

    dummy_model = InstationaryModelIP(                 
        **building_blocks,
    )
    u_delta, percentage = construct_noise_data(model = dummy_model, 
                                               q_exact = q_exact,
                                               noise_level = setup['noise_level'],
                                               product=products['bochner_prod_V'],
                                               time_depend_noise=True)

    
    ############################### Cost ###############################

    if setup['observation_operator']['name'] == 'identity':
        C_mat = pd2.SparseMatrix()
        material_model.assemble_observation_operator_matrix(
            C_mat,
            'identity'
        )
        C = DealIIMatrixOperator(matrix = C_mat)        
        print(C)
    else:
        raise ValueError

    y_delta = C.apply(u_delta)

    assert (len(y_delta) == setup['nt'] + 1)
    assert (y_delta.space == C.range) 

    logger.debug(f'noise percentage is {percentage:3.4e}')
    logger.debug(f'noise_level is {setup["noise_level"]:3.4e}')

    y_delta = y_delta[1:]
    constant_cost_term = y_delta.pairwise_inner(y_delta, product=products['prod_C'])

    # #print(products['prod_C'].assemble().matrix)

    # linear_cost_term = NumpyMatrixOperator(
    #     matrix = np.arange(135*135).reshape((135,135))
    # )
    

    # print(linear_cost_term.apply(u_delta))
    

    # # linear_cost_term_mat = products['prod_C'].apply(y_delta)
    # # linear_cost_term_mat = C.apply_adjoint(linear_cost_term_mat)
    # # print(linear_cost_term_mat)

    # import sys
    # sys.exit()




    

    # linear_cost_term = DealIIMatrixOperator(
    #     matrix = linear_cost_mat
    # )
    # bilinear_cost_term = DealIIMatrixOperator(
    #     matrix = C_mat_T @ products['prod_C'].assemble().matrix @ C_mat
    # )

    print(products['prod_C'].matrix.m())
    print(C.matrix.m())

    bilinear_cost_mat = pd2.SparseMatrix(C.matrix.get_sparsity_pattern())
    print(products['prod_C'].matrix.m())
    print(C.matrix.m())

    products['prod_C'].matrix.mmult(bilinear_cost_mat, C.matrix)

    #C.matrix.Tmmult(bilinear_cost_mat, bilinear_cost_mat)    
    
    bilinear_cost_term = DealIIMatrixOperator(
        matrix = bilinear_cost_mat
    )
    print(bilinear_cost_term)
    import sys
    sys.exit()

    ############################### Final ###############################

    building_blocks['constant_cost_term'] = constant_cost_term
    building_blocks['linear_cost_term'] = linear_cost_term
    building_blocks['bilinear_cost_term'] = bilinear_cost_term
    building_blocks['model_constants'] = {
        'A_coercivity_constant_estimator' : A_coercivity_constant_estimator,
        'C_continuity_constant' : C_continuity_constant
    }

    return building_blocks


    ############################### Final ###############################
    

    
    return building_blocks

if __name__ == "__main__":
    par_dim = 2

    T_initial = 0
    T_final = 1
    nt = 50
    delta_t = (T_final - T_initial) / nt


    assert T_final > T_initial
    q_circ = 3*np.ones((1, par_dim))
    
    buf = np.zeros((1, par_dim))
    buf[10:50] = 1
    q_exact = buf + q_circ

    bounds = np.zeros((par_dim, 2))
    bounds[:,0] = 0.001
    bounds[:,1] = 1e3

    setup = {
        'T_initial': T_initial,                       # Start time of the simulation
        'T_final': T_final,                           # End time of the simulation
        'delta_t': delta_t,                           # Time step size
        'nt': nt,                                     # Number of time steps
        'N' : None,
        'par_dim' : 2,
        'noise_percentage': None,                     # Relative noise level, will be set by 'build_InstationaryModelIP'
        'noise_level': 1e-5,                          # Absolute noise magnitude added to data
        'q_circ': q_circ,                             # Backgroundlevel for the parameter
        'q_exact_function': None,                     # Exact parameter as function, will be set by 'build_InstationaryModelIP'
        'q_exact': q_exact,                           # Exact parameter values, will be set by 'build_InstationaryModelIP'
        'q_time_dep': False,                          # Whether parameter is time-dependent (bool)
        'riesz_rep_grad': True,                       # Use Riesz representative for gradient in optimization
        'bounds': bounds,                             # Bounds on parameter values (e.g., for optimization)
        'products': {                                 # Inner products used in the problem
            'prod_H': 'l2',                           # Product on H_h
            'prod_Q': 'euclid',                       # Product on Q_h
            'prod_V': 'h1_0_semi',                    # Product on V_h
            'prod_C': 'l2',                           # Product on C_h
            'bochner_prod_Q': 'bochner_euclid',       # Product on Q_h^K
            'bochner_prod_V': 'bochner_h1_0_semi'     # Product on V_h^K
        },
        'observation_operator': {
            'name': 'identity',                       # Type of observation operator (e.g., identity = full state observed)
        },
        'time_stepper' : {
            'name' : 'newman_second_order',
            'zeta' : 0.5
        }
    }




    build_InstationaryModelIP(
        setup = setup,
        logger=None
    ) 