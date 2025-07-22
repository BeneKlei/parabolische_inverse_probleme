# ~~~
# This file is part of the paper:
#
#           " Adaptive Trust Region Reduced Basis Methods for Inverse Parameter Identification Problems "
#
#   https://github.com/michikartmann
#
# Copyright 2023 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Contributors team: Michael Kartmann, Tim Keil
# ~~~
# Description:
# This file prepares the analytical PDE problem which gets handed with to the discretizer.

import numpy as np
import scipy
import logging
import sys
import inspect

from typing import Dict, Tuple

import pymor.models.basic as InstationaryProblem
from pymor.basic import *
from pymor.analyticalproblems.functions import ProductFunction
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.discretizers.builtin import discretize_instationary_cg
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.discretizers.builtin.grids.rect import RectGrid
from pymor.parameters.base import Mu

from .utils import thermal_block_problem_h1, twodhatfunction
from RBInvParam.model import InstationaryModelIP
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.utils.discretization import split_constant_and_parameterized_operator, construct_noise_data
from RBInvParam.evaluators import UnAssembledA, UnAssembledB
from RBInvParam.products import BochnerProductOperator
from RBInvParam.model import InstationaryModelIP
from RBInvParam.error_estimator import CoercivityConstantEstimator

def whole_problem(N : int = 100,
                  contrast_parameter : int = 2, 
                  parameter_location : str = 'diffusion', 
                  boundary_conditions : str = 'dirichlet', 
                  exact_parameter : str = 'PacMan', 
                  parameter_elements : str = 'P1',
                  time_factor: str = 'constant',
                  T_final : int = 1) -> Tuple:
    
    # check input and set problem type
    assert parameter_location in {'diffusion', 'reaction' }, 'Change parameter location to "diffusion" or "dirichlet"'
    assert boundary_conditions in {'dirichlet', 'robin' }, 'Change boundary conditions to "dirichlet" or "robin"'
    assert exact_parameter in {'PacMan', 'Kirchner', 'dummy', 'other' }, 'Change exact parameter to "Landweber" or "other"'
    assert time_factor in {'constant', 'sinus'}
    assert parameter_elements in {'P1' }, ' "P1" '
    
    problem_type = parameter_location + ' ' + boundary_conditions + ' ' + exact_parameter + ' ' + parameter_elements
    p = thermal_block_problem_h1((N, N))    
    f = ConstantFunction(1, 2)                                                  # PDE rhs f
    
    # define diffusion and reaction parameter coefficients
    if parameter_location == 'diffusion':
        diffusion = p.diffusion.with_(name='')                                 
        reaction = None 
    else:
        reaction = p.diffusion.with_(name='')
        diffusion = ConstantFunction(1, 2)
    
    # define boundary conditions
    if boundary_conditions == 'dirichlet':
        domain = RectDomain([[0., 0.], [1., 1.]],
                            bottom='dirichlet', left='dirichlet',
                            right='dirichlet', top='dirichlet')
        dirichlet_data = ConstantFunction(0, 2)
        robin_data = None
    else:
        domain = RectDomain([[0., 0.], [1., 1.]],
                            bottom='robin', left='robin',
                            right='robin', top='robin')
        u_out = ConstantFunction(1, 2)                                         
        robin_data = (ConstantFunction(1, 2), u_out)
        dirichlet_data = None
        
    # define pyMOR analytical problem

    stationary_problem = StationaryProblem(domain = domain,
                                           diffusion = diffusion,
                                           reaction = reaction,
                                           rhs = f,
                                           robin_data = robin_data,
                                           dirichlet_data = dirichlet_data)
    

    background = ConstantFunction(3, 2)
    # define exact parameter 
    if exact_parameter == 'PacMan':      
        # Note:
        # Exact parameter from the paper [A Reduced Basis Landweber method for nonlinear inverse problems]
        # by D. Garmatter, B. Haasdonk, B. Harrach, 2016.
        
        ccc  = 1
        omega_1_1 = ExpressionFunction('(5/30. < x[0]) * (x[0] < 9/30.) \
                                       * (3/30. < x[1]) * (x[1] < 27/30.)', 2)
        omega_1_2 = ExpressionFunction('(9/30. < x[0]) * (x[0] < 27/30.) \
                                       * (3/30. < x[1]) * (x[1] < 7/30.)', 2)
        omega_1_3 = ExpressionFunction('(9/30. < x[0]) * (x[0] < 27/30.) \
                                       * (23/30. < x[1]) * (x[1] < 27/30.)', 2)
        omega_2 = ExpressionFunction('sqrt((x[0]-18/30.)**2 \
                                     + (x[1]-15/30.)**2) <= 4/30.', 2)
        q_exact_function = ccc * contrast_parameter * (omega_1_1 + omega_1_2 + omega_1_3) - 2 * omega_2
                    
    elif exact_parameter == 'Kirchner':  
         
        # Note:
        # Exact parameter from the dissertation [Adaptive regularization and discretization for nonlin-
        # ear inverse problems with PDEs] by A. Kirchner, 2014.
        
         ccc = 1
         q_1 = ExpressionFunction('1/(2*pi*0.01)*exp(-0.5*((2*x[0]-0.5)/0.1)**2 - 0.5*((2*x[1]-0.5)/0.1)**2)', 2 )  
         q_2 = ExpressionFunction('1/(2*pi*0.01)*exp(-0.5*((0.8*x[0]-0.5)/0.1)**2 - 0.5*((0.8*x[1]-0.5)/0.1)**2)', 2 ) 
         q_exact_function =  ccc*q_1 +  ccc*q_2
    elif exact_parameter == 'dummy':
         q_exact_function =  ConstantFunction(1, 2)      
    elif exact_parameter == 'other':
        
        multiscale_part = ConstantFunction(0,2)
        twodhat = twodhatfunction([[0.6,0.75,0.9], [0.1,0.25,0.4]])             
        continuous_part = GenericFunction(twodhat, 2)
        upper_right = ExpressionFunction('(0.2 < x[0]) * (x[0] < 0.3) \
                                       * (0.7< x[1]) * (x[1] < 0.8)', 2)  
        discontinuous_part = ExpressionFunction('sqrt((x[0]-0.25)**2 \
                                     + (x[1]-0.25)**2) <= 0.1', 2)
        smooth_part =  ExpressionFunction('exp(-20*(x[0]-0.75)**2 - 20*(x[1]-0.75)**2)', 2 )
        sinus_background = ConstantFunction(0,2)
        q_exact_function = smooth_part + discontinuous_part + continuous_part + upper_right + multiscale_part + sinus_background #+ middle_part

    if time_factor == 'constant':
        pass
    elif time_factor == 'sinus':
        time_factor = ExpressionFunction(
            #expression = 'sin(2*pi*t)[0]',
            expression = 'sin(pi*t)[0]',
            dim_domain = 2,
            parameters = {'t' : 1}
        )
        
        q_exact_function = ProductFunction(
            functions = [time_factor, q_exact_function]
        )
    else:
        raise ValueError
    
    q_exact_function = q_exact_function + background
        
    # create exact model with exact parameter and energy_product model
    if parameter_location == 'diffusion':
        
        # problem for simulating the exact data
        exact_analytical_problem = StationaryProblem(
                                    domain = domain,
                                    diffusion = q_exact_function,
                                    reaction = reaction,
                                    rhs = f,
                                    robin_data = robin_data,
                                    dirichlet_data = dirichlet_data
                                    )
        
        # problem for assembling energy product corresp to q = 1
        energy_problem = StationaryProblem(
                                    domain = domain,
                                    diffusion = ConstantFunction(1, 2),
                                    reaction = None,
                                    rhs = f,
                                    robin_data = robin_data,
                                    dirichlet_data = dirichlet_data
                                    )
        
    else:
        
        # problem for computing u_exact data
        exact_analytical_problem = StationaryProblem(
                                    domain = domain,
                                    diffusion = diffusion,
                                    reaction = q_exact_function,
                                    rhs = f,
                                    robin_data = robin_data,
                                    dirichlet_data = dirichlet_data
                                    )
        
        # problem for assembling energy product corresp to q = 1
        energy_problem = StationaryProblem(
                                    domain = domain,
                                    diffusion = ConstantFunction(1, 2),
                                    reaction = ConstantFunction(1, 2),
                                    rhs = f,
                                    robin_data = robin_data,
                                    dirichlet_data = dirichlet_data
                                    )
    



    # # get exact parameter evaluated on rectangular mesh
    # discretized_domain, _ = discretize_domain_default(domain, np.sqrt(2)/N, RectGrid)
    # xp = discretized_domain.centers(2)
    # q_exact = q_exact_function(xp)

    initial_data = ConstantFunction(0, 2)

    analytical_problem = InstationaryProblem(
        stationary_part = stationary_problem,
        initial_data = initial_data,
        T = T_final,
        name = 'Instationary_' + stationary_problem.name
    )

    exact_analytical_problem = InstationaryProblem(
        stationary_part = exact_analytical_problem,
        initial_data = initial_data,
        T = T_final,
        name = 'Instationary_' + exact_analytical_problem.name
    )

    energy_problem = InstationaryProblem(
        stationary_part = energy_problem,
        initial_data = initial_data,
        T = T_final,
        name = 'Instationary_' + energy_problem.name
    )

    return analytical_problem, q_exact_function, problem_type, exact_analytical_problem, energy_problem


def discretize_instationary_IP(analytical_problem : InstationaryProblem, 
                               setup : Dict,
                               logger: logging.Logger = None) -> Dict:

    if logger:
        logger = logger
    else:
        logger = get_default_logger(inspect.getframeinfo(inspect.currentframe()).function)
        logger.setLevel(logging.DEBUG)              
                            
    ############################### FOM ###############################
    primal_fom, grid_data = discretize_instationary_cg(analytical_problem,
                                                       diameter=setup['dims']['diameter'],
                                                       preassemble= False,
                                                       grid_type = RectGrid,
                                                       nt = setup['dims']['nt'])
                                                             
    Q_h = NumpyVectorSpace(dim = setup['dims']['par_dim'])
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
        'h1' : primal_fom.products['h1'].assemble().matrix,
        'h1_semi' : primal_fom.products['h1_semi'].assemble().matrix,
        'l2' : primal_fom.products['l2'].assemble().matrix,
        'h1_0' : primal_fom.products['h1_0'].assemble().matrix,
        'h1_0_semi' : primal_fom.products['h1_0_semi'].assemble().matrix,
        'l2_0' : primal_fom.products['l2_0'].assemble().matrix
    }

    # process products dict
    product_names = {}
    product_name = ''
    for (key,value) in setup['model_parameter']['products'].items():
        buf = value.split('_')
        if buf[0] == 'bochner':
            product_name = '_'.join(buf[1:])
        else:
            product_name = value
        assert product_name in assembled_products.keys()
        product_names[key] = product_name

    products['prod_H'] = NumpyMatrixOperator(
        matrix = assembled_products[product_names['prod_H']]
    )

    products['prod_Q'] = NumpyMatrixOperator(
        matrix = assembled_products[product_names['prod_Q']]
    )
    products['prod_V'] = NumpyMatrixOperator(
        matrix = assembled_products[product_names['prod_V']]
    )

    products['prod_C'] = NumpyMatrixOperator(
        matrix = assembled_products[product_names['prod_C']]
    )


    products['bochner_prod_Q'] = BochnerProductOperator(
        product=NumpyMatrixOperator(
            matrix = assembled_products[product_names['bochner_prod_Q']]
        ),
        delta_t=setup['model_parameter']['delta_t'],
        space = Q_h,
        nt = setup['dims']['nt']
    )

    products['bochner_prod_V'] = BochnerProductOperator(
        product=NumpyMatrixOperator(
            matrix = assembled_products[product_names['bochner_prod_V']]
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
        grid = grid_data['grid'],
        boundary_info = grid_data['boundary_info'],
        source=V_h,
        range=V_h,
        Q = Q_h
    )
    
    B = UnAssembledB(
        reaction_problem = ('reaction' in setup['model_parameter']['problem_type']),
        grid = grid_data['grid'],
        boundary_info = grid_data['boundary_info'],
        source=Q_h,
        range=V_h,
        Q = Q_h,
        V = V_h
    )

    ############################### Coercivity ###############################

            
    problem_type = setup['model_parameter']['problem_type']
    
    assert product_names['prod_V'] == 'h1_0_semi'
    
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
        matrix = products['prod_Q'].matrix.T @ q_circ.to_numpy().T
    )
    bilinear_reg_term = NumpyMatrixOperator(
        matrix = products['prod_Q'].matrix
    )

    ############################### Dummy Model ###############################

    # # get exact parameter evaluated on rectangular mesh
    xp = grid_data['grid'].centers(2)
    q_exact_function = setup['model_parameter']['q_exact_function'] 
    
    if setup['problem_parameter']['time_factor'] != 'constant':
        assert setup['model_parameter']['q_time_dep']
        ts = [setup['model_parameter']['T_initial'] + (i + 1) * setup['model_parameter']['delta_t']            
            for i in range(setup['dims']['nt'])
        ]
        parameters = q_exact_function.parameters

        setup['model_parameter']['q_exact'] = np.array([q_exact_function.evaluate(
            xp, mu = parameters.parse({'t' : t})
        ) for t in ts])
    else:
        if setup['model_parameter']['q_time_dep']:
            setup['model_parameter']['q_exact'] = np.array([q_exact_function.evaluate(
                xp
            ) for _ in range(setup['dims']['nt'])])
        else:
            setup['model_parameter']['q_exact'] = np.array([q_exact_function.evaluate(
                xp
            )])

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
        'setup' : setup,
        'bounds' : setup['model_parameter']['bounds']
    }

    dummy_model = InstationaryModelIP(                 
        **building_blocks,
    )

    ############################### Cost ###############################

    if setup['model_parameter']['observation_operator']['name'] == 'identity':
        C = NumpyMatrixOperator(
            scipy.sparse.identity(V_h.dim)
        )
        C_continuity_constant = 1.0
    elif setup['model_parameter']['observation_operator']['name'] == 'RoI':

        assert 'RoI' in setup['model_parameter']['observation_operator'].keys()
        assert isinstance(setup['model_parameter']['observation_operator']['RoI'], np.ndarray)
        assert setup['model_parameter']['observation_operator']['RoI'].shape == (2,2)

        RoI = setup['model_parameter']['observation_operator']['RoI']


        centers = grid_data['grid'].centers(codim=2)
        mask = (RoI[0,0] <= centers) & (centers <= RoI[0,1]) & (RoI[1,0] <= centers) & (centers <=  RoI[1,1])
        mask = np.all(mask, axis=1)
        mask = np.logical_not(mask)
        idxes = np.argwhere(mask)

        C = scipy.sparse.identity(V_h.dim)
        C = C.tolil()
        C[idxes, idxes] = 0
        C = C.tocsr()

        C = NumpyMatrixOperator(
            C
        )
        C_continuity_constant = 1.0     
    else:
        raise ValueError

        
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
        matrix = C.matrix.T @ products['prod_C'].assemble().matrix @ y_delta.to_numpy().T
    )
    bilinear_cost_term = NumpyMatrixOperator(
        matrix = C.matrix.T @ products['prod_C'].assemble().matrix @ C.matrix
    )

    ############################### Final ###############################

    building_blocks['constant_cost_term'] = constant_cost_term
    building_blocks['linear_cost_term'] = linear_cost_term
    building_blocks['bilinear_cost_term'] = bilinear_cost_term
    building_blocks['model_constants'] = {
        'A_coercivity_constant_estimator' : A_coercivity_constant_estimator,
        'C_continuity_constant' : C_continuity_constant
    }

    return building_blocks, grid_data, assembled_products

def build_InstationaryModelIP(setup : Dict,
                              logger : logging.Logger = None) -> Tuple[InstationaryModelIP, Dict]:
    
    if not logger:
        logger = get_default_logger(logger_name=sys._getframe().f_code.co_name)

    logger.debug('Construct problem..')                                                     
    analytical_problem, q_exact_function, problem_type, exact_analytical_problem, energy_problem = \
        whole_problem(**setup['problem_parameter'])
    
    setup['model_parameter']['problem_type'] = problem_type
    setup['model_parameter']['parameters'] = analytical_problem.parameters
    setup['model_parameter']['q_exact_function'] = q_exact_function
    
    logger.debug('Discretizing problem...')                
    building_blocks, grid_data, assembled_products = discretize_instationary_IP(analytical_problem, setup)
    
    return InstationaryModelIP(**building_blocks), grid_data, assembled_products