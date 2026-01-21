import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
import inspect
from typing import Dict

import RBInvParam.problems.hyperelasticity.hyperelasticity_model as hm

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
    hyperelasticity_model_config.delta_t = setup['delta_t']
    hyperelasticity_model_config.spatial_resolution = setup['spatial_resolution']

    hyperelasticity_model_config.se_type = setup['stored_energy']['type']
    if setup['stored_energy']['hyperparameter']: 
        hyperelasticity_model_config.se_hyperparameter = setup['stored_energy']['hyperparameter']

    hyperelasticity_model_config.body_force_type = setup['body_force']['type']
    if setup['body_force']['hyperparameter']: 
        hyperelasticity_model_config.body_force_hyperparameter = setup['body_force']['hyperparameter']

    hyperelasticity_model = hm.HyperElasticityModel(hyperelasticity_model_config)

    import sys
    sys.exit()
    hyperelasticity_model.make_state_grid()
    hyperelasticity_model.make_param_grid()
    hyperelasticity_model.setup_system()

    import sys
    sys.exit()

    return build_InstationaryModelIP(
        setup = setup,
        material_model = hyperelasticity_model,
        A_class = HyperElasticitiyFOMEvaluatorA,
        logger = logger
    )

    