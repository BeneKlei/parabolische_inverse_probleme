import numpy as np
import logging
import sys
import argparse
from pathlib import Path

from pymor.basic import *

from RBInvParam.optimizer import *
from RBInvParam.utils.io import save_dict_to_pkl
from RBInvParam.utils.logger import get_default_logger, reset_logger

from RBInvParam.problems.problems import build_InstationaryModelIP

def run_optimization(
    setup: Dict,
    optimizer_parameter: Dict,
    save_path: Path) -> None:

    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    method = optimizer_parameter['method']
    assert method in ['FOM_IRGNM', 'Qr_IRGNM', 'TR_IRGNM']
    assert save_path.exists()

    ####################################### SETUP LOGGER #######################################
    logfile_path= save_path / f'{method}.log'
    logger = get_default_logger(logger_name=sys._getframe().f_code.co_name,
                                logfile_path=logfile_path, 
                                use_timestemp=False)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    set_log_levels({
        'pymor' : 'WARN'
    })

    set_defaults({})

    ####################################### SETUP FOM #######################################

    FOM = build_InstationaryModelIP(setup, logger)

    logger.info(f"Dumping model setup to {save_path / 'setup.pkl'}.")
    save_dict_to_pkl(path=save_path / 'setup.pkl', 
                     data = setup,
                     use_timestamp=False)

    q_exact = FOM.setup['model_parameter']['q_exact']
    
    ####################################### SETUP OPTIMIZER #######################################

    if method == 'FOM_IRGNM':
        optimizer = FOMOptimizer(
            FOM = FOM,
            optimizer_parameter = optimizer_parameter,
            logger = logger,
            save_path = save_path
        )

    elif method == 'Qr_IRGNM':
        optimizer = QrFOMOptimizer(
            FOM = FOM,
            optimizer_parameter = optimizer_parameter,
            logger = logger,
            save_path = save_path
        )

    elif method == 'TR_IRGNM':
        optimizer = QrVrROMOptimizer(
            FOM = FOM,
            optimizer_parameter = optimizer_parameter,
            logger = logger,
            save_path = save_path
        )
    else:
        # Should never happend
        raise ValueError
    
    ####################################### RUN & FINALIZE #######################################

    q_est = optimizer.solve()

    logger.debug("Differnce to q_exact:")
    logger.debug("L^inf") 
    delta_q = q_est - q_exact
    logger.debug(f"  {np.max(np.abs(delta_q.to_numpy())):3.4e}")
    
    if setup['model_parameter']['q_time_dep']:
        norm_delta_q = np.sqrt(FOM.products['bochner_prod_Q'].apply2(delta_q, delta_q))[0,0]
        norm_q_exact = np.sqrt(FOM.products['bochner_prod_Q'].apply2(q_exact, q_exact))[0,0]
    else:
        norm_delta_q = np.sqrt(FOM.products['prod_Q'].apply2(delta_q, delta_q))[0,0]
        norm_q_exact = np.sqrt(FOM.products['prod_Q'].apply2(q_exact, q_exact))[0,0]
    
    logger.debug(f"  Absolute error: {norm_delta_q:3.4e}")
    logger.debug(f"  Relative error: {norm_delta_q / norm_q_exact * 100:3.4}%.")
    
    # TODO make this in a better way
    reset_logger(logger.name)
    reset_logger('build_InstationaryModelIP')
    reset_logger('discretize_instationary_IP')
    reset_logger('FOMOptimizer')
    reset_logger('QrFOMOptimizer')
    reset_logger('QrVrROMOptimizer')
    reset_logger('InstationaryModelIP')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run optimization for a given setup and optimizer parameters.')
    parser.add_argument(
        '--setup',
        type=str,
        required=True,
        help='Path to the setup configuration file (in JSON or YAML format).'
    )
    parser.add_argument(
        '--optimizer-parameter',
        type=str,
        required=True,
        help='Path to the optimizer parameters configuration file (in JSON or YAML format).'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        required=True,
        help='Directory where results and logs will be saved.'
    )
    args = parser.parse_args()
    
    run_optimization(setup = args.setup, 
                     optimizer_parameter = args.optimizer_parameter, 
                     save_path = args.save_path)