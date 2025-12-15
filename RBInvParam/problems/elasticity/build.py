import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import scipy
import logging
import inspect
from typing import Dict

from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
 
import RBInvParam.problems.shared.material_model as mm
import RBInvParam.problems.elasticity.elasticity_model as em

from RBInvParam.problems.shared.pymor_dealii_bindings.vectorarray import DealIIVectorSpace
from RBInvParam.problems.shared.pymor_dealii_bindings.operator import DealIIMatrixOperator, DealIISymmetricMatrixOperator
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.utils.discretization import construct_noise_data, process_product_names
from RBInvParam.model import InstationaryModelIP
from RBInvParam.products import BochnerProductOperator, EnergyProductOperator
from RBInvParam.error_estimators.objective_error_estimators import CoercivityConstantEstimator

from RBInvParam.problems.elasticity.evaluators import ElasticitiyFOMEvaluatorA, ElasticitiyFOMEvaluatorB
#from utils import * 

def build_InstationaryModelIP(setup : Dict,
                              logger : logging.Logger = None) -> InstationaryModelIP:

    if logger:
        logger = logger
    else:
        logger = get_default_logger(inspect.getframeinfo(inspect.currentframe()).function)
        logger.setLevel(logging.DEBUG)   

    logger.debug('Construct problem..')

    material_model_config = mm.MaterialModelConfig()
    material_model_config.nt = setup['dims']['nt']
    material_model_config.T_initial = setup['T_initial']
    material_model_config.T_final = setup['T_final']
    material_model_config.delta_t = setup['delta_t']
    material_model_config.spatial_resolution = setup['spatial_resolution']

    material_model_config.system_matrix_type = setup['system_matrix']['type']
    if setup['system_matrix']['hyperparameter']: 
        material_model_config.system_matrix_hyperparameter = setup['system_matrix']['hyperparameter']

    material_model_config.body_force_type = setup['body_force']['type']
    if setup['body_force']['hyperparameter']: 
        material_model_config.body_force_hyperparameter = setup['body_force']['hyperparameter']

    elasticity_model = em.ElasticityModel(material_model_config)
    elasticity_model.make_grid()
    elasticity_model.setup_system()

    ############################### State and Param Space ###############################

    setup['dims']['par_dim'] = elasticity_model.param_space_dim
    setup['dims']['state_dim'] = elasticity_model.state_space_dim

    Q_h = NumpyVectorSpace(dim = setup['dims']['par_dim'])
    V_h = DealIIVectorSpace(dim = setup['dims']['state_dim'])

    ############################### State and Param Products ###############################

    product_names = setup['products']

    _str_to_enum_map_state = {
        'l2' : mm.StateProductType.L2, 
        'l2_0' : mm.StateProductType.L2_0, 
        'h1_semi' : mm.StateProductType.H1_semi, 
        'h1_0_semi' : mm.StateProductType.H1_0_semi,
        'h1' : mm.StateProductType.H1, 
        'h1_0' : mm.StateProductType.H1_0, 
    }

    elasticity_model.assemble_product_H(_str_to_enum_map_state[product_names['prod_H']])
    elasticity_model.assemble_product_V(_str_to_enum_map_state[product_names['prod_V']])

    products = {
        'prod_H' : None,
        'prod_Q' : None,
        'prod_V' : None,
        'prod_C' : None,
        'energy' : None,
        'bochner_prod_Q' : None,
        'bochner_prod_V' : None,
        'bochner_prod_C' : None,
        'bochner_energy' : None,
    }

    assembled_parameter_products  = {
        'euclid' : scipy.sparse.identity(Q_h.dim)
    }

    products['L2'] = DealIIMatrixOperator(
        matrix = elasticity_model.product_L2
    )

    products['H1'] = DealIIMatrixOperator(
        matrix = elasticity_model.product_H1
    )

    products['prod_H'] = DealIIMatrixOperator(
        matrix = elasticity_model.product_H
    )

    products['prod_Q'] = NumpyMatrixOperator(
        matrix = assembled_parameter_products[product_names['prod_Q']]
    )

    products['prod_V'] = DealIIMatrixOperator(
        matrix = elasticity_model.product_V
    )

    products['energy'] = EnergyProductOperator(
        kinetic_product = products['prod_H'],
        potential_product = products['prod_V'],
        space = V_h
    )

    products['bochner_prod_Q'] = BochnerProductOperator(
        product=NumpyMatrixOperator(
            matrix = assembled_parameter_products[product_names['prod_Q']]
        ),
        delta_t=setup['delta_t'],
        space = Q_h,
        nt = setup['dims']['nt']
    )

    products['bochner_prod_V'] = BochnerProductOperator(
        product=DealIIMatrixOperator(
            matrix = elasticity_model.product_V
        ),
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
    
    elasticity_model.assemble_mass_matrix()
    M = DealIISymmetricMatrixOperator(
        matrix = elasticity_model.mass_matrix
    )
    L = V_h.make_array(elasticity_model.force_list)

    A = ElasticitiyFOMEvaluatorA(
        material_model = material_model,
        source = V_h,
        range = V_h,
        Q = Q_h,
        parameter_names = ['lambda', 'mu']
    )
    B = ElasticitiyFOMEvaluatorB(
        material_model = material_model,
        source=Q_h,
        range=V_h,
        Q = Q_h,
        V = V_h   
    )
    ############################### Coercivity ###############################

    assert product_names['prod_V'] == 'h1_0_semi'
    # I AM NOT SURE THAT THIS IS CORRECT! JUST FOR TESTING
    #A_coercivity_constant_estimator_function = lambda q: 1

    x = np.min([
        2 * setup['system_matrix']['hyperparameter']['mu'],
        2 * setup['system_matrix']['hyperparameter']['nu'],
        2 * setup['system_matrix']['hyperparameter']['mu'] + \
        3 * setup['system_matrix']['hyperparameter']['lambda']
    ])

    #A_coercivity_constant_estimator_function = lambda q: np.min(q.to_numpy()) * x
    y = np.min(setup['bounds'][:,0])
    assert y > 0
    A_coercivity_constant_estimator_function = lambda q: y * x
    
    

    A_coercivity_constant_estimator = CoercivityConstantEstimator(
        coercivity_estimator_function = A_coercivity_constant_estimator_function,
        Q = Q_h,
        q_time_dep = setup['q_time_dep']
    )

    ############################### Regularization ###############################

    q_circ = setup['q_circ']
    assert type(q_circ) == np.ndarray
    q_circ = Q_h.make_array(q_circ)
    assert len(q_circ) in [setup['dims']['nt']+1, 1]

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
    assert len(q_exact) in [setup['dims']['nt'] + 1, 1]

    building_blocks = {
        'initial_data' : initial_data, 
        'M' : M,
        'A' : A,
        'L' : L,
        'B' : B,
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
    elasticity_model.assemble_observation_operator_matrix(
        setup['observation_operator']['type'],
        setup['observation_operator']['hyperparameter']
    )
    C = DealIIMatrixOperator(matrix = elasticity_model.observation_operator)
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

    setup['dims']['observation_space_dim'] = elasticity_model.observation_space_dim
    C_h = DealIIVectorSpace(dim = setup['dims']['observation_space_dim'])
    
    elasticity_model.assemble_product_C(_str_to_enum_map_observation_space[product_names['prod_C']])

    products['prod_C'] = DealIIMatrixOperator(
        matrix = elasticity_model.product_C
    )

    products['bochner_prod_C'] = BochnerProductOperator(
        product=DealIIMatrixOperator(
            matrix = elasticity_model.product_C
        ),
        delta_t=setup['delta_t'],
        space = C_h,
        nt = setup['dims']['nt']
    )        
    # --------------------------------------------------------------------

    y_delta, percentage = construct_noise_data(model = dummy_model, 
                                               q_exact = q_exact,
                                               C = C,
                                               noise_level = setup['noise_level'],
                                               product=products['bochner_prod_C'],
                                               time_depend_noise=True)
    

    dummy_model.A.elasticity_model.save_time_series(
        [v.real_part.impl for v in y_delta.vectors],
        str('y_delta'),
        str(setup['save_path']),
        np.linspace(dummy_model.T_initial, dummy_model.T_final, dummy_model.nt+1)
    )

    # diff_y = y_delta - dummy_model.solve_state(q_exact)
    # dummy_model.A.elasticity_model.save_time_series(
    #     [v.real_part.impl for v in diff_y.vectors],
    #     str('diff_y'),
    #     str(setup['save_path']),
    #     np.linspace(dummy_model.T_initial, dummy_model.T_final, dummy_model.nt+1)
    # )

    assert (len(y_delta) == setup['dims']['nt'] + 1)
    assert (y_delta.space == C.range) 

    logger.debug(f'noise percentage is {percentage:3.4e}')
    logger.debug(f'noise_level is {setup["noise_level"]:3.4e}')    

    #--------------------------------------------------------
    constant_cost_term = y_delta.pairwise_inner(y_delta, product=products['prod_C'])
    #--------------------------------------------------------
    linear_cost_term = products['prod_C'].apply(y_delta)
    linear_cost_term = C.apply_adjoint(linear_cost_term)
    #--------------------------------------------------------   
    elasticity_model.assemble_bilinear_cost_matrix()

    bilinear_cost_term = DealIIMatrixOperator(
        matrix = elasticity_model.bilinear_cost_operator
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
