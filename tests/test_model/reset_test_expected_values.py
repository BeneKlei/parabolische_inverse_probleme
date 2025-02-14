import numpy as np
import argparse
import sys
import importlib.util
from typing import Dict, List
from pathlib import Path

from pymor.basic import set_log_levels
from pymor.core.pickle import dump
from pymor.vectorarrays.interface import VectorArray

from RBInvParam.problems.problems import build_InstationaryModelIP
from RBInvParam.utils.logger import get_default_logger

CWD = Path(__file__).parent.resolve()
spec = importlib.util.spec_from_file_location('setups',  CWD / '../setups.py')
setups = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setups)
SETUPS = setups.SETUPS

set_log_levels({
    'pymor' : 'WARN'
})


parser = argparse.ArgumentParser(description='Generate a set of values expected from InstationaryModelIP')
parser.add_argument('--setup_name',type=str,help='The name of the configuration to load.')
args = parser.parse_args()

def reset_test_expected_values(setup_name: str = None,
                               qs: List[VectorArray] = None,
                               ds: List[VectorArray] = None,
                               alphas : List = None,
                               num_samples: int = 10,
                               seed : int = 42) -> None:
    data_given = (qs is not None) or (ds is not None)

    if setup_name:
        assert setup_name in SETUPS
        setup_names = [setup_name]
    else:
        setup_names = list(SETUPS.keys())

    for setup_name in setup_names:
        reset_params = True

        logger = get_default_logger()
                    
        path = CWD / Path(f'{setup_name}.pkl')
        if path.exists():
            logger.error(f"Found existing solutions for {setup_name}. Delete before redefining!")
            sys.exit(-1)

        setup = SETUPS[setup_name]
        FOM = build_InstationaryModelIP(setup, logger) 

        if data_given:
            assert np.all([q in FOM.Q for q in qs])
            assert np.all([d in FOM.Q for d in ds])
            assert len(qs) == len(ds) == len(alphas)
            reset_params = False

        if reset_params:
            np.random.seed(seed)
            if setup['model_parameter']['q_time_dep']:                                                 
                x = setup['dims']['nt']
            else:
                x = 1
            qs = [FOM.Q.make_array(np.random.random((x, setup['dims']['par_dim']))) \
                for _ in range(num_samples)]
            ds = [FOM.Q.make_array(np.random.random((x, setup['dims']['par_dim']))) \
                for _ in range(num_samples)]
            alphas = list(np.random.random(num_samples))

        us = []
        ps = []
        lin_us = []
        lin_ps = []
        Js = []
        nabla_Js = []
        lin_Js = []
        nabla_lin_Js = []
        
        for idx in range(len(qs)): 
            logger.info(f"Generating sample {idx}")
            u = FOM.solve_state(qs[idx])
            p = FOM.solve_adjoint(qs[idx], u)
            lin_u = FOM.solve_linearized_state(qs[idx], ds[idx], u)
            lin_p = FOM.solve_linearized_adjoint(qs[idx], u, lin_u)
            J = FOM.objective(u, qs[idx], alpha=0)
            nabla_J = FOM.gradient(u, p, qs[idx], alpha=0)
            lin_J = FOM.linearized_objective(qs[idx], ds[idx], u, lin_u, alphas[idx])
            nabla_lin_J = FOM.linearized_gradient(qs[idx], ds[idx], u, lin_p, alphas[idx])
        
            us.append(u)        
            ps.append(p)
            lin_us.append(lin_u)
            lin_ps.append(lin_p)                        
            Js.append(J)
            nabla_Js.append(nabla_J)                        
            lin_Js.append(lin_J)
            nabla_lin_Js.append(nabla_lin_J)

        expected_values = {
            'qs' : qs,
            'ds' : ds,
            'alphas' : alphas,
            'us' : us,
            'ps' : ps,
            'lin_us' : lin_us,
            'lin_ps' : lin_ps,
            'Js' : Js,
            'nabla_Js' : nabla_Js,
            'lin_Js' : lin_Js,
            'nabla_lin_Js' : nabla_lin_Js
        }

        print(f"Dump expected values to {path.as_posix()}")
        with open(path, 'wb') as file:
            dump(expected_values, file)

    
if __name__ == '__main__':
    reset_test_expected_values(setup_name = args.setup_name)