import numpy as np
import logging
from pathlib import Path
from typing import Dict, Union
from datetime import datetime

from pymor.core.pickle import dump

from RBInvParam.model import InstationaryModelIP
from RBInvParam.problems.problems import whole_problem
from RBInvParam.discretizer import discretize_instationary_IP

def save_dict_to_pkl(path: Union[str, Path],
                     data: Dict) -> None:

    path = Path(path)
    assert path.suffix in ['.pkl', 'pickle']
    assert path.parent.exists()

    assert isinstance(data, Dict)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_with_timestamp = f"{timestamp}_{path.stem}{path.suffix}"
    path_with_timestamp = path.parent / filename_with_timestamp
    
    with open(path_with_timestamp, 'wb') as file:
        dump(data, file)

def load_FOM_from_config(config : Dict,
                         logger: logging.Logger = None) -> InstationaryModelIP:

    analytical_problem, q_exact, N, problem_type, _, _ = whole_problem(**config['problem_parameter'])
    config['model_parameter']['parameters'] = analytical_problem.parameters
    
    if config['model_parameter']['q_time_dep']:                                                 
        config['model_parameter']['q_exact'] = np.array([q_exact for _ in range(config['dims']['nt'])])
    else:
        config['model_parameter']['q_exact'] = np.array([q_exact])

    building_blocks = discretize_instationary_IP(analytical_problem,
                                                 config['model_parameter'],
                                                 config['dims'], 
                                                 problem_type,
                                                 logger=logger) 
    return InstationaryModelIP(                 
        *building_blocks,
        dims = config['dims'],
        model_parameter = config['model_parameter']
    )