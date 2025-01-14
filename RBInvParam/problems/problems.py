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
import logging
from typing import Dict, Tuple

from pymor.basic import *
from pymor.analyticalproblems.instationary import InstationaryProblem

from RBInvParam.problems.utils import thermal_block_problem_h1, twodhatfunction
from RBInvParam.model import InstationaryModelIP
from RBInvParam.discretizer import discretize_instationary_IP
from RBInvParam.utils.logger import get_default_logger


def whole_problem(N = 100, contrast_parameter = 2, parameter_location = 'diffusion', boundary_conditions = 'dirichlet', exact_parameter = 'PacMan', parameter_elements = 'P1'):    
    # check input and set problem type
    assert parameter_location in {'diffusion', 'reaction' }, 'Change parameter location to "diffusion" or "dirichlet"'
    assert boundary_conditions in {'dirichlet', 'robin' }, 'Change boundary conditions to "dirichlet" or "robin"'
    assert exact_parameter in {'PacMan', 'Kirchner', 'dummy', 'other' }, 'Change exact parameter to "Landweber" or "other"'
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

    problem = StationaryProblem(
                                domain = domain,
                                diffusion = diffusion,
                                reaction = reaction,
                                rhs = f,
                                robin_data = robin_data,
                                dirichlet_data = dirichlet_data
                                )
    
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
        exact_q_function = ConstantFunction(3, 2) +\
                    ccc * contrast_parameter * (omega_1_1 + omega_1_2 + omega_1_3) - 2 * omega_2
                    
    elif exact_parameter == 'Kirchner':  
         
        # Note:
        # Exact parameter from the dissertation [Adaptive regularization and discretization for nonlin-
        # ear inverse problems with PDEs] by A. Kirchner, 2014.
        
         ccc = 1
         q_1 = ExpressionFunction('1/(2*pi*0.01)*exp(-0.5*((2*x[0]-0.5)/0.1)**2 - 0.5*((2*x[1]-0.5)/0.1)**2)', 2 )  
         q_2 = ExpressionFunction('1/(2*pi*0.01)*exp(-0.5*((0.8*x[0]-0.5)/0.1)**2 - 0.5*((0.8*x[1]-0.5)/0.1)**2)', 2 ) 
         exact_q_function =  ccc*q_1 +  ccc*q_2 + ConstantFunction(3, 2)
    elif exact_parameter == 'dummy':
         exact_q_function =  ConstantFunction(1, 2)
         
    elif exact_parameter == 'other':
        
        multiscale_part = ConstantFunction(0,2)
        twodhat = twodhatfunction([[0.6,0.75,0.9], [0.1,0.25,0.4]])             
        continuous_part = GenericFunction(twodhat, 2)
        upper_right = ExpressionFunction('(0.2 < x[0]) * (x[0] < 0.3) \
                                       * (0.7< x[1]) * (x[1] < 0.8)', 2)  
        discontinuous_part = ExpressionFunction('sqrt((x[0]-0.25)**2 \
                                     + (x[1]-0.25)**2) <= 0.1', 2)
        smooth_part =  ExpressionFunction('exp(-20*(x[0]-0.75)**2 - 20*(x[1]-0.75)**2)', 2 )
        background = ConstantFunction(3, 2)
        sinus_background = ConstantFunction(0,2)
        exact_q_function = background + smooth_part + discontinuous_part + continuous_part + upper_right + multiscale_part + sinus_background #+ middle_part
        
    # create exact model with exact parameter and energy_product model
    if parameter_location == 'diffusion':
        
        # problem for simulating the exact data
        exact_analytical_problem = StationaryProblem(
                                    domain = domain,
                                    diffusion = exact_q_function,
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
                                    reaction = exact_q_function,
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
    



    # get exact parameter evaluated on rectangular mesh
    discretized_domain, _ = discretize_domain_default(domain, np.sqrt(2)/N, RectGrid)
    xp = discretized_domain.centers(2)
    q_exact = exact_q_function(xp)

    initial_data = ConstantFunction(0, 2)
    T = 1

    problem = InstationaryProblem(
        stationary_part = problem,
        initial_data = initial_data,
        T = T,
        name = 'Instationary_' + problem.name
    )

    exact_analytical_problem = InstationaryProblem(
        stationary_part = exact_analytical_problem,
        initial_data = initial_data,
        T = T,
        name = 'Instationary_' + exact_analytical_problem.name
    )

    energy_problem = InstationaryProblem(
        stationary_part = energy_problem,
        initial_data = initial_data,
        T = T,
        name = 'Instationary_' + energy_problem.name
    )

    return problem, q_exact, N, problem_type, exact_analytical_problem, energy_problem



def build_InstationaryModelIP(setup : Dict,
                              logger : logging.Logger = None) -> Tuple[Dict,InstationaryModelIP]:
    
    if not logger:
        logger = get_default_logger()

    logger.debug('Construct problem..')                                                     
    analytical_problem, q_exact, N, problem_type, exact_analytical_problem, energy_problem = whole_problem(**setup['problem_parameter'])
    

    setup['model_parameter']['parameters'] = analytical_problem.parameters
    if setup['model_parameter']['q_time_dep']:                                                 
        setup['model_parameter']['q_exact'] = np.array([q_exact for _ in range(setup['dims']['nt'])])
    else:
        setup['model_parameter']['q_exact'] = np.array([q_exact])
        
    logger.debug('Discretizing problem...')                                                
    building_blocks = discretize_instationary_IP(analytical_problem,
                                                 setup['model_parameter'],
                                                 setup['dims'], 
                                                 problem_type)
     
    setup['model_parameter']['q_exact'] = building_blocks['Q'].make_array(setup['model_parameter']['q_exact'])
    building_blocks['setup'] = setup

    return (setup, InstationaryModelIP(                 
        **building_blocks,
    ))