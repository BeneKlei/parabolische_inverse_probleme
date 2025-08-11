import numpy as np
import scipy
import logging
import sys
import inspect

from typing import Dict, Tuple

from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.analyticalproblems.functions import ConstantFunction
 
import pymor_dealii_bindings as pd2

import RBInvParam.problems.elasticity.material_model as mm

from RBInvParam.problems.elasticity.pymor_dealii_bindings.vectorarray import DealIIVectorSpace
from RBInvParam.problems.elasticity.pymor_dealii_bindings.operator import DealIIMatrixOperator
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.model import InstationaryModelIP
from RBInvParam.products import BochnerProductOperator



def build_InstationaryModelIP(setup : Dict,
                              logger : logging.Logger = None) -> Tuple[InstationaryModelIP, Dict]:

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

    assembled_products = {
        'h1' : pd2.SparseMatrix(),
        'h1_semi' : pd2.SparseMatrix(),
        'l2' : pd2.SparseMatrix(),
        'h1_0' : pd2.SparseMatrix(),
        'h1_0_semi' : pd2.SparseMatrix(),
        'l2_0' : pd2.SparseMatrix(),
    }

    material_model.assemble_h1_matrix(assembled_products['h1'])
    material_model.assemble_h1_semi_matrix(assembled_products['h1_semi'])
    material_model.assemble_l2_matrix(assembled_products['l2'])
    material_model.assemble_h1_0_matrix(assembled_products['h1_0'])
    material_model.assemble_h1_0_semi_matrix(assembled_products['h1_0_semi'])
    material_model.assemble_l2_0_matrix(assembled_products['l2_0'])

    #process products dict
    product_names = {}
    product_name = ''
    for (key,value) in setup['products'].items():
        buf = value.split('_')
        if buf[0] == 'bochner':
            product_name = '_'.join(buf[1:])
        else:
            product_name = value
        assert product_name in assembled_products.keys()
        product_names[key] = product_name

    products['prod_H'] = DealIIMatrixOperator(
        matrix = assembled_products[product_names['prod_H']]
    )

    products['prod_Q'] = DealIIMatrixOperator(
        matrix = assembled_products[product_names['prod_Q']]
    )
    products['prod_V'] = DealIIMatrixOperator(
        matrix = assembled_products[product_names['prod_V']]
    )

    products['prod_C'] = DealIIMatrixOperator(
        matrix = assembled_products[product_names['prod_C']]
    )

    # products['bochner_prod_Q'] = BochnerProductOperator(
    #     product=DealIIMatrixOperator(
    #         matrix = assembled_products[product_names['bochner_prod_Q']]
    #     ),
    #     delta_t=setup['delta_t'],
    #     space = Q_h,
    #     nt = setup['nt']
    # )
    products['bochner_prod_Q'] = None

    products['bochner_prod_V'] = BochnerProductOperator(
        product=DealIIMatrixOperator(
            matrix = assembled_products[product_names['bochner_prod_V']]
        ),
        delta_t=setup['delta_t'],
        space = V_h,
        nt = setup['nt']
    )

    ############################### Operators ###############################

    #initial_data = ConstantFunction(0, 3)
    u_0 = V_h.zeros()

    M = pd2.SparseMatrix()
    material_model.assemble_mass_matrix(M)

    L = np.array(np.asarray(l) for l in material_model.get_force_list())

    import sys
    sys.exit()

    ############################### Coercivity ###############################

    ############################### Regularization ###############################

    q_circ = setup['q_circ']
    assert type(q_circ) == np.ndarray
    q_circ = Q_h.make_array(q_circ)
    assert len(q_circ) in [setup['nt'], 1]

    constant_reg_term = q_circ.pairwise_inner(q_circ, product=products['prod_Q'])    
    linear_reg_term = DealIIMatrixOperator(
        matrix = products['prod_Q'].matrix.T @ q_circ.to_numpy().T
    )
    bilinear_reg_term = DealIIMatrixOperator(
        matrix = products['prod_Q'].matrix
    )

    ############################### Final ###############################
    

    building_blocks = {
        'u_0' : u_0, 
        'M' : M,
        'A' : None,
        'L' : L,
        'B' : None,
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
    return building_blocks

if __name__ == "__main__":
    par_dim = 2

    T_initial = 0
    T_final = 1
    nt = 2
    delta_t = (T_final - T_initial) / nt


    assert T_final > T_initial
    q_circ = 3*np.ones((1, par_dim))
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
        'q_exact': None,                              # Exact parameter values, will be set by 'build_InstationaryModelIP'
        'q_time_dep': False,                          # Whether parameter is time-dependent (bool)
        'riesz_rep_grad': True,                       # Use Riesz representative for gradient in optimization
        'bounds': bounds,                             # Bounds on parameter values (e.g., for optimization)
        'products': {                                 # Inner products used in the problem
            'prod_H': 'l2',                           # Product on H_h
            'prod_Q': 'h1',                           # Product on Q_h
            'prod_V': 'h1_0_semi',                    # Product on V_h
            'prod_C': 'l2',                           # Product on C_h
            'bochner_prod_Q': 'bochner_h1',           # Product on Q_h^K
            'bochner_prod_V': 'bochner_h1_0_semi'     # Product on V_h^K
        },
        'observation_operator': {
            'name': 'identity',                       # Type of observation operator (e.g., identity = full state observed)
        }
    }




    build_InstationaryModelIP(
        setup = setup,
        logger=None
    ) 