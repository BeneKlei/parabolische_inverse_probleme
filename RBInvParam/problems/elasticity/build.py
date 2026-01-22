import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
import inspect
from typing import Dict

import RBInvParam.problems.elasticity.elasticity_model as em

from RBInvParam.problems.shared.pymor_dealii_bindings.operator import *
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.model import InstationaryModelIP

from RBInvParam.problems.shared.build import build_InstationaryModelIP
from RBInvParam.problems.elasticity.evaluators import ElasticitiyFOMEvaluatorA

def build_ElasticityModelIP(setup : Dict,
                            logger : logging.Logger = None) -> InstationaryModelIP:

    if logger:
        logger = logger
    else:
        logger = get_default_logger(inspect.getframeinfo(inspect.currentframe()).function)
        logger.setLevel(logging.DEBUG)   

    logger.debug('Constructing problem..')

    elasticity_model_config = em.ElasticityModelConfig()
    elasticity_model_config.nt = setup['dims']['nt']
    elasticity_model_config.T_initial = setup['T_initial']
    elasticity_model_config.T_final = setup['T_final']
    elasticity_model_config.delta_t = setup['delta_t']
    elasticity_model_config.spatial_resolution = setup['spatial_resolution']

    elasticity_model_config.material_operator_type = setup['material_operator']['type']
    if setup['material_operator']['hyperparameter']: 
        elasticity_model_config.material_operator_hyperparameter = setup['material_operator']['hyperparameter']

    elasticity_model_config.body_force_type = setup['body_force']['type']
    if setup['body_force']['hyperparameter']: 
        elasticity_model_config.body_force_hyperparameter = setup['body_force']['hyperparameter']

    elasticity_model = em.ElasticityModel(elasticity_model_config)
    elasticity_model.make_param_grid()
    elasticity_model.make_state_grid()
    elasticity_model.setup_system()
    
    ############################### Coercivity ###############################

    assert setup['products']['prod_V'] == 'h1_0_semi'
    # I AM NOT SURE THAT THIS IS CORRECT! JUST FOR TESTING
    #A_coercivity_constant_estimator_function = lambda q: 1

    x = np.min([
        2 * setup['material_operator']['hyperparameter']['mu'],
        2 * setup['material_operator']['hyperparameter']['nu'],
        2 * setup['material_operator']['hyperparameter']['mu'] + \
        3 * setup['material_operator']['hyperparameter']['lambda']
    ])

    y = np.min(setup['bounds'][:,0])
    assert y > 0
    coercivity_constant_estimator_function = lambda q: y * x

    ##########################################################################

    return build_InstationaryModelIP(
        setup = setup,
        material_model = elasticity_model,
        EvaluatorA_class = ElasticitiyFOMEvaluatorA,
        coercivity_constant_estimator_function = coercivity_constant_estimator_function,
        logger = logger
    )

