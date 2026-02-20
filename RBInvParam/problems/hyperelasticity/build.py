import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
import inspect
from typing import Dict

import RBInvParam.problems.hyperelasticity.hyperelasticity_model as hm
#import material_model as mm
#import hyperelasticity_model as hm

from RBInvParam.problems.shared.pymor_dealii_bindings.operator import *
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.model import InstationaryModelIP

from RBInvParam.problems.shared.build import build_InstationaryModelIP
from RBInvParam.problems.hyperelasticity.evaluators import HyperElasticitiyFOMEvaluatorA

def build_HyperElasticityModelIP(setup : Dict,
                                 logger : logging.Logger = None) -> InstationaryModelIP:

    if logger:
        logger = logger
    else:
        logger = get_default_logger(inspect.getframeinfo(inspect.currentframe()).function)
        logger.setLevel(logging.DEBUG)   

    logger.debug('Constructing problem..')

    hyperelasticity_model_config = hm.HyperElasticityModelConfig()
    hyperelasticity_model_config.nt = setup['dims']['nt']
    hyperelasticity_model_config.T_initial = setup['T_initial']
    hyperelasticity_model_config.T_final = setup['T_final']
    hyperelasticity_model_config.p1 = setup['p1']
    hyperelasticity_model_config.p2 = setup['p2']
    hyperelasticity_model_config.param_grid_resolution = setup['param_grid_resolution']
    hyperelasticity_model_config.state_grid_resolution = setup['state_grid_resolution']

    hyperelasticity_model_config.se_type = setup['stored_energy']['type']
    if setup['stored_energy']['hyperparameter']: 
        hyperelasticity_model_config.se_hyperparameter = setup['stored_energy']['hyperparameter']

    hyperelasticity_model_config.body_force_type = setup['body_force']['type']
    if setup['body_force']['hyperparameter']: 
        hyperelasticity_model_config.body_force_hyperparameter = setup['body_force']['hyperparameter']

    hyperelasticity_model_config.BC_type = setup['boundary_condition']['type']
    if setup['boundary_condition']['hyperparameter']: 
        hyperelasticity_model_config.BC_hyperparameter = setup['boundary_condition']['hyperparameter']

    hyperelasticity_model = hm.HyperElasticityModel(hyperelasticity_model_config)
    hyperelasticity_model.setup_system()

    ############################### Coercivity ###############################

    assert setup['products']['prod_V'] == 'h1_0_semi'
    # I AM NOT SURE THAT THIS IS CORRECT! JUST FOR TESTING
    coercivity_constant_estimator_function = lambda q: 1

    return build_InstationaryModelIP(
        setup = setup,
        material_model = hyperelasticity_model,
        EvaluatorA_class = HyperElasticitiyFOMEvaluatorA,
        coercivity_constant_estimator_function = coercivity_constant_estimator_function,
        logger = logger
    )

    
