import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import scipy
import logging
from typing import Dict, Callable

from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
 
import RBInvParam.problems.shared.material_model as mm
#import material_model as mm

from RBInvParam.problems.shared.pymor_dealii_bindings.vectorarray import DealIIVectorSpace
from RBInvParam.problems.shared.pymor_dealii_bindings.operator import *

from RBInvParam.utils.discretization import construct_noise_data
from RBInvParam.model import InstationaryModelIP
from RBInvParam.products import BochnerProductOperator, EnergyProductOperator
from RBInvParam.error_estimators.objective_error_estimators import CoercivityConstantEstimator
from RBInvParam.evaluators import EvaluatorA


def build_InstationaryModelIP(setup : Dict,
                              material_model: mm.MaterialModel,
                              EvaluatorA_class: EvaluatorA,
                              coercivity_constant_estimator_function: Callable[[np.ndarray], float],
                              logger : logging.Logger = None) -> InstationaryModelIP:

    setup['delta_t'] = material_model.delta_t

    ############################### State and Param Space ###############################

    setup['dims']['par_dim'] = material_model.param_space_dim
    setup['dims']['state_dim'] = material_model.state_space_dim

    Q_h = NumpyVectorSpace(dim = setup['dims']['par_dim'])
    V_h = DealIIVectorSpace(dim = setup['dims']['state_dim'])

    ############################### State and Param Products ###############################

    product_names = setup['products']

    _str_to_enum_map_state = {
        'euclid'    : mm.FEProductType.EUCLID, 
        'l2'        : mm.FEProductType.L2, 
        'l2_0'      : mm.FEProductType.L2_0, 
        'h1_semi'   : mm.FEProductType.H1_semi, 
        'h1_0_semi' : mm.FEProductType.H1_0_semi,
        'h1'        : mm.FEProductType.H1, 
        'h1_0'      : mm.FEProductType.H1_0, 
    }

    #material_model.assemble_product_H(_str_to_enum_map_state[product_names['prod_H']])
    #material_model.assemble_product_V(_str_to_enum_map_state[product_names['prod_V']])
    

    products = {
        'prod_H' : None,
        'prod_Q' : None,
        'prod_V' : None,
        'prod_C' : None,
        'prod_reg' : None,
        'energy' : None,
        'bochner_prod_Q' : None,
        'bochner_prod_V' : None,
        'bochner_prod_C' : None,
        'bochner_energy' : None,
    }

    # assembled_parameter_products  = {
    #     'euclid' : scipy.sparse.identity(Q_h.dim)
    # }

    products['L2'] = SparseMatrixOperator(
        op = material_model.assemble_state_product_op(
            mm.FEProductType.L2
        )
    )

    products['H1'] = SparseMatrixOperator(
        op = material_model.assemble_state_product_op(
            mm.FEProductType.H1
        )
    )

    products['prod_H'] = SparseMatrixOperator(
        op = material_model.assemble_state_product_op(
            _str_to_enum_map_state[product_names['prod_H']]
        )
    )

    products['prod_Q'] = NumpyDealIISparseMatrixOperator(
        op = material_model.assemble_param_product_op(
            _str_to_enum_map_state[product_names['prod_Q']]
        )
    )

    products['prod_V'] = SparseMatrixOperator(
        op = material_model.assemble_state_product_op(
            _str_to_enum_map_state[product_names['prod_V']]
        )
    )

    products['prod_reg'] = NumpyDealIISparseMatrixOperator(
        op = material_model.assemble_param_product_op(
            _str_to_enum_map_state[product_names['prod_reg']]
        )
    )

    products['energy'] = EnergyProductOperator(
        kinetic_product = products['prod_H'],
        potential_product = products['prod_V'],
        space = V_h
    )

    products['bochner_prod_Q'] = BochnerProductOperator(
        product=products['prod_Q'],
        delta_t=setup['delta_t'],
        space = Q_h,
        nt = setup['dims']['nt']
    )

    products['bochner_prod_V'] = BochnerProductOperator(
        product=products['prod_V'],
        delta_t=setup['delta_t'],
        space = V_h,
        nt = setup['dims']['nt']
    )

    products['bochner_energy'] = BochnerProductOperator(
        product=products['energy'],
        delta_t=setup['delta_t'],
        space = V_h,
        nt = setup['dims']['nt']
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
        'second_adjoint' : {
            'zeroth_order' : zero_data,
            'first_order' : zero_data
        },
    }
    
    # material_model.assemble_mass_matrix()    
    # matrix = pd2.SparseMatrix(material_model.mass_matrix.get_sparsity_pattern())
    # matrix.copy_from(material_model.mass_matrix)    
    
    M = SparseMatrixOperator(
        op = material_model.assemble_mass_op()    
    )

    L = V_h.make_array(material_model.force_list)

    A = EvaluatorA_class(
        material_model,
        source = V_h,
        range = V_h,
        Q = Q_h    
    )  
    # B = ElasticitiyFOMEvaluatorB(
    #     elasticity_model = elasticity_model,
    #     source=Q_h,
    #     range=V_h,
    #     Q = Q_h,
    #     V = V_h   
    # )
    
    A_coercivity_constant_estimator = CoercivityConstantEstimator(
        coercivity_estimator_function = coercivity_constant_estimator_function,
        Q = Q_h,
        q_time_dep = setup['q_time_dep']
    )

    ############################### Regularization ###############################

    q_circ = setup['q_circ']
    assert type(q_circ) == np.ndarray
    q_circ = Q_h.make_array(q_circ)
    assert len(q_circ) in [setup['dims']['nt']+1, 1]

    prod_reg_op = products['prod_reg']

    constant_reg_term = q_circ.pairwise_inner(q_circ, product=prod_reg_op)
    linear_vec = prod_reg_op.apply_adjoint(q_circ)

    linear_reg_term = NumpyMatrixOperator(
        matrix=linear_vec.to_numpy().reshape(-1, 1)
    )

    bilinear_reg_term = prod_reg_op    

    # constant_reg_term = q_circ.pairwise_inner(q_circ, product=products['prod_Q'])    
    # linear_reg_term = NumpyMatrixOperator(
    #     matrix = products['prod_Q'].matrix.T @ q_circ.to_numpy().T
    # )
    # bilinear_reg_term = NumpyMatrixOperator(
    #     matrix = products['prod_Q'].matrix
    # )

    ############################### u^delta / Dummy Model ###############################
    q_exact = setup['q_exact']
    assert type(q_exact) == np.ndarray
    q_exact = Q_h.make_array(q_exact)
    assert len(q_exact) in [setup['dims']['nt'] + 1, 1]

    building_blocks = {
        'initial_data' : initial_data, 
        'M' : M,
        'A' : A,
        'L' : L,
        'C' : None,
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

    ############################### Cost ###############################
    
    # material_model.assemble_observation_operator_matrix(
    #     setup['observation_operator']['type'],
    #     setup['observation_operator']['hyperparameter']
    # )
    # C = DealIIMatrixOperator(matrix = material_model.observation_operator)

    C = SparseMatrixOperator(
        op = material_model.assemble_observation_op(
            setup['observation_operator']['type'],
            setup['observation_operator']['hyperparameter']
        )
    )
    
    C_continuity_constant = 1.0
    building_blocks['C'] = C

    # -------------------------------------------------------------------- 
    _str_to_enum_map_observation_space = {
        'euclid' : mm.ObservationSpaceProductType.EUCLID, 
        'state_l2' : mm.ObservationSpaceProductType.STATE_L2, 
        'state_l2_0' : mm.ObservationSpaceProductType.STATE_L2_0, 
        'state_h1_semi' : mm.ObservationSpaceProductType.STATE_H1_semi, 
        'state_h1_0_semi' : mm.ObservationSpaceProductType.STATE_H1_0_semi,
        'state_h1' : mm.ObservationSpaceProductType.STATE_H1, 
        'state_h1_0' : mm.ObservationSpaceProductType.STATE_H1_0, 
    }

    setup['dims']['observation_space_dim'] = material_model.observation_space_dim
    C_h = DealIIVectorSpace(dim = setup['dims']['observation_space_dim'])
    
    # material_model.assemble_product_C(_str_to_enum_map_observation_space[product_names['prod_C']])

    # products['prod_C'] = DealIIMatrixOperator(
    #     matrix = material_model.product_C
    # )

    products['prod_C'] = SparseMatrixOperator(
        op = material_model.assemble_product_C_op(
            _str_to_enum_map_observation_space[product_names['prod_C']]
        )
    )

    products['bochner_prod_C'] = BochnerProductOperator(
        product=products['prod_C'],
        delta_t=setup['delta_t'],
        space = C_h,
        nt = setup['dims']['nt']
    )   

    # --------------------------------------------------------------------

    y_delta, u_exact, noise_info = construct_noise_data(
        model=dummy_model,
        q_exact=q_exact,
        C=C,
        noise_level=setup['noise_info']['noise_level_input'],
        noise_level_mode=setup['noise_info']['noise_level_mode'],  
        product=products['bochner_prod_C'],
        time_depend_noise=True,
    )

    setup['y_delta'] = y_delta.to_numpy()
    setup['noise_info'] = noise_info
    
    assert (len(y_delta) == setup['dims']['nt'] + 1)
    assert (y_delta.space == C.range) 

    y = C.apply(u_exact)
    diff_y = y_delta - y
    norm_diff_y = np.sqrt(
        dummy_model.products['bochner_prod_C'].apply2(diff_y, diff_y)
    )[0, 0]

    rel_noise_level_y = norm_diff_y / np.sqrt(
        dummy_model.products['bochner_prod_C'].apply2(y, y)
    )[0, 0]

    rel_noise_level_u = norm_diff_y / np.sqrt(
        dummy_model.products['bochner_prod_V'].apply2(u_exact, u_exact)
    )[0, 0]

    abs_noise_level = norm_diff_y

    logger.debug(f'noise_level_input = {setup["noise_info"]["noise_level_input"]:3.4e}')
    logger.debug(f'noise_level_mode  = {setup["noise_info"]["noise_level_mode"]}')
    logger.debug(f'abs_noise_level   = {abs_noise_level:3.4e}')
    logger.debug(f'rel_noise_level_y = {rel_noise_level_y:3.4e}')
    logger.debug(f'rel_noise_level_u = {rel_noise_level_u:3.4e}')

    #--------------------------------------------------------
    constant_cost_term = y_delta.pairwise_inner(y_delta, product=products['prod_C'])
    #--------------------------------------------------------
    linear_cost_term = products['prod_C'].apply(y_delta)
    linear_cost_term = C.apply_adjoint(linear_cost_term)
    #--------------------------------------------------------   

    bilinear_cost_term = SparseMatrixOperator(
        op = material_model.assemble_bilinear_cost_op(
            C.op,
            products['prod_C'].op
        )
    )

    ############################### Final ###############################

    building_blocks['constant_cost_term'] = constant_cost_term
    building_blocks['linear_cost_term'] = linear_cost_term
    building_blocks['bilinear_cost_term'] = bilinear_cost_term
    building_blocks['model_constants'] = None
    building_blocks['model_constants'] = {
        'A_coercivity_constant_estimator' : A_coercivity_constant_estimator,
        'C_continuity_constant' : C_continuity_constant,
    }

    return InstationaryModelIP(
        **building_blocks,
        logger= logger
    )


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
        'y_delta' : None,
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

    FOM = InstationaryModelIP(**build_InstationaryModelIP(
        setup = setup,
        logger=None
    )) 
