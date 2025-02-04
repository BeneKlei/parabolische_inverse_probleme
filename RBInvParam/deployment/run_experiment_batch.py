import argparse
import logging
import importlib.util
import os
import sys
import shutil
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List

from RBInvParam.utils.logger import get_default_logger
from RBInvParam.utils.io import save_dict_to_pkl
from RBInvParam.deployment.run_optimization import run_optimization

ALLOWED_TARGETS = ['local', 'ag-server', 'palma']
#PALMA_SCRIPT_PATH = Path(__file__).parent.resolve() / "local_test.sh"
PALMA_SCRIPT_PATH = Path(__file__).parent.resolve() / "queue_experiment.sh"

def _run_experiment_batch_local(working_dir: Path,
                                experiments: List,
                                logger: logging.Logger):

    for (experiment_name, experiment_config) in experiments.items():
        logger.info(f'Starting experiment {experiment_name}.')
        try:
            setup, optimizer_parameter = experiment_config
            save_path = working_dir / experiment_name

            if save_path.exists():
                logger.error(f'For experiment {experiment_name} a result directory already exist. SKIPPING it.')    
                continue
            else:
                os.mkdir(save_path)


            run_optimization(
                setup = setup,
                optimizer_parameter = optimizer_parameter,
                save_path = save_path
            )

    
        except Exception as e:
            logger.error(f'An error occured during experiment {experiment_name}:')
            traceback.print_exc()
            logger.info('Continuing with next experiment.')
            continue

        logger.info(f'Experiment {experiment_name} has finished.')
        logger.info(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    
    logger.info(f'All experiments have finished.')

def _run_experiment_batch_palma(working_dir: Path,
                                experiments: List,
                                logger: logging.Logger):

    experiment_jobs = []

    for (experiment_name, experiment_config) in experiments.items():
        logger.info(f'Queueing experiment {experiment_name}.')
        try:
            setup, optimizer_parameter = experiment_config
            save_path = working_dir / experiment_name

            if save_path.exists():
                logger.error(f'For experiment {experiment_name} a result directory already exist. SKIPPING it.')    
                continue
            else:
                os.mkdir(save_path)

                temp_setup_path = save_path / 'setup.pkl'
                save_dict_to_pkl(
                    path = temp_setup_path,
                    data = setup,
                    use_timestamp = False
                )

                temp_optimizer_parameter_path = save_path / 'optimizer_parameter.pkl'
                save_dict_to_pkl(
                    path = temp_optimizer_parameter_path,
                    data = optimizer_parameter,
                    use_timestamp = False
                )

            cmd = [
                'sbatch',
                '--job-name=' + experiment_name,
                PALMA_SCRIPT_PATH,
                temp_setup_path.as_posix(),
                temp_optimizer_parameter_path.as_posix(),
                save_path.as_posix()
            ]

            experiment_job = subprocess.Popen(cmd)
            experiment_jobs.append(experiment_job)
        
        except Exception as e:
            logger.error(f'An error occured during queueing experiment {experiment_name}:')
            traceback.print_exc()
            logger.info('Continuing with next experiment.')
            continue
    
    for experiment_job in experiment_jobs:
        experiment_job.wait()
    

def run_experiment_batch(
    experiments: Path,
    target : str,
    working_dir : Path = None) -> None:

    assert experiments.parent.exists()

    if not working_dir:
        working_dir = experiments.parent

    ####################################### SETUP LOGGER #######################################

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile_path= working_dir / f'experiment_run_{timestamp}.log'
    logger = get_default_logger(logger_name=sys._getframe().f_code.co_name,
                                logfile_path=logfile_path, 
                                use_timestemp=False)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.info(f'Loading experiments from {experiments}')

    try:
        spec = importlib.util.spec_from_file_location('experiments', experiments)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except:
        logger.error(f'Can not load {experiments}. Stopping run.')
        return 

    if hasattr(module, 'EXPERIMENTS'):
        experiments = module.EXPERIMENTS        
        experiment_names = list(experiments.keys())
    else:
        logger.error(f'Can not load EXPERIMENTS from {experiments}. Stopping run.')
        return 
    
    if target in ALLOWED_TARGETS:
        logger.info(f'Using target {target} to run experiments on.')
    else:
        logger.error(f'Given target {target} is unknown. Please use a target in {ALLOWED_TARGETS}')


    logger.info(f'Found {len(experiment_names)} experiments:')
    for experiment_name in experiment_names:
        logger.info(f'  - {experiment_name}')

    
    dir_exists = []
    remove = False
    for experiment_name in experiment_names:
        save_path = working_dir / experiment_name
        if save_path.exists():
            dir_exists.append(save_path)
    
    if len(dir_exists): 
        logger.info('Found existing result directories for the following experiments:')
        for experiment_paths in dir_exists:
            logger.info(f'  - {experiment_paths.parts[-1]}')
        

        logger.info('If you not remove them the respective experiments will NOT be run.')
        logger.info('Do you want to remove them (Y/n):')
        response = input().strip().lower()

        if response in ('y', 'yes', ''):
            logger.info("Removing the results.")
            remove = True
        elif response in ('n', 'no'):
            logger.info("Not removing the results. SKIPPING the respective experiments.")   
        else:
            logger.info("Response is undefined. Stopping execution.")
            return 
        
    if remove:
        for dir in dir_exists:
            shutil.rmtree(dir)

    if target in ['local', 'ag-server']:
        _run_experiment_batch_local(
            working_dir = working_dir,
            experiments = experiments,
            logger = logger
        )
    else:
        _run_experiment_batch_palma(
            working_dir = working_dir,
            experiments = experiments,
            logger = logger
        )

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a batch of experiments.')
    parser.add_argument('experiments', type=Path, help='Path to the experiments file')
    parser.add_argument('target', type=str, choices=ALLOWED_TARGETS, help='Target where to run the experiments')
    parser.add_argument('--working_dir', type=Path, default=None, help='Optional working directory for the experiments')
    args = parser.parse_args()

    run_experiment_batch(args.experiments, args.target, args.working_dir)
    
    