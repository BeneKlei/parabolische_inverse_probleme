import pytest
import importlib.util
from pathlib import Path
from typing import Dict

from pymor.basic import *
from pymor.tools.floatcmp import float_cmp_all
from pymor.core.pickle import load

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

ABS_TOL = 1e-14
REL_TOL = 1e-14

logger = get_default_logger()

test_configs_u = []
test_configs_p = []
test_configs_lin_u = []
test_configs_lin_p = []
test_configs_J = []
test_configs_nabla_J = []
test_configs_lin_J = []
test_configs_nabla_lin_J = []

for setup_name, setup in SETUPS.items():        
    FOM = build_InstationaryModelIP(setup, logger) 

    path = CWD / Path(f'./{setup_name}.pkl')
    assert path.exists()
    with open(path, 'rb') as file:
        solutions = load(file)

    test_config = {
        'setup_name' : setup_name, 
        'model' : FOM,
        'qs' : solutions['qs'],
        'ds' : solutions['ds'],
        'alphas' : solutions['alphas'],
        'us' : solutions['us'],
        'ps' : solutions['ps'],
        'lin_us' : solutions['lin_us'],
        'lin_ps' : solutions['lin_ps'],
        'Js' : solutions['Js'],
        'nabla_Js' : solutions['nabla_Js'],
        'lin_Js' : solutions['lin_Js'],
        'nabla_lin_Js' : solutions['nabla_lin_Js'],
    }


    test_configs_u.append(test_config.copy())
    test_configs_p.append(test_config.copy())
    test_configs_lin_u.append(test_config.copy())
    test_configs_lin_p.append(test_config.copy())
    test_configs_J.append(test_config.copy())
    test_configs_nabla_J.append(test_config.copy())
    test_configs_lin_J.append(test_config.copy())
    test_configs_nabla_lin_J.append(test_config.copy())

    
@pytest.mark.parametrize("config", test_configs_u, ids=[config['setup_name'] for config in test_configs_u])
def test_us(config: Dict) -> None:
    qs = config['qs']
    us = config['us']
    model = config['model']
    for idx in range(len(qs)):
        assert float_cmp_all(us[idx].to_numpy(), model.solve_state(qs[idx]).to_numpy(), REL_TOL, ABS_TOL)

@pytest.mark.parametrize("config", test_configs_p, ids=[config['setup_name'] for config in test_configs_p])
def test_ps(config: Dict) -> None:
    qs = config['qs']
    us = config['us']
    ps = config['ps']
    model = config['model']
    for idx in range(len(qs)):
        assert float_cmp_all(ps[idx].to_numpy(), model.solve_adjoint(qs[idx], us[idx]).to_numpy(), REL_TOL, ABS_TOL)

@pytest.mark.parametrize("config", test_configs_lin_u, ids=[config['setup_name'] for config in test_configs_lin_u])
def test_lin_us(config: Dict) -> None:
    qs = config['qs']
    ds = config['ds']
    us = config['us']
    lin_us = config['lin_us']
    model = config['model']
    for idx in range(len(qs)):
        assert float_cmp_all(lin_us[idx].to_numpy(), model.solve_linearized_state(qs[idx], ds[idx], us[idx]).to_numpy(), REL_TOL, ABS_TOL)

@pytest.mark.parametrize("config", test_configs_lin_p, ids=[config['setup_name'] for config in test_configs_lin_p])
def test_lin_ps(config: Dict) -> None:
    qs = config['qs']
    us = config['us']
    lin_us = config['lin_us']
    lin_ps = config['lin_ps']
    model = config['model']
    for idx in range(len(qs)):
        assert float_cmp_all(lin_ps[idx].to_numpy(), model.solve_linearized_adjoint(qs[idx], us[idx], lin_us[idx]).to_numpy(), REL_TOL, ABS_TOL)

@pytest.mark.parametrize("config", test_configs_J, ids=[config['setup_name'] for config in test_configs_J])
def test_Js(config: Dict) -> None:
    qs = config['qs']
    us = config['us']
    Js = config['Js']
    model = config['model']
    for idx in range(len(qs)):
        assert float_cmp_all(Js[idx], model.objective(us[idx], qs[idx], alpha=0), REL_TOL, ABS_TOL)

@pytest.mark.parametrize("config", test_configs_nabla_J, ids=[config['setup_name'] for config in test_configs_nabla_J])
def test_nabla_Js(config: Dict) -> None:
    qs = config['qs']
    us = config['us']
    ps = config['ps']
    nabla_Js = config['nabla_Js']
    model = config['model']
    for idx in range(len(qs)):
        assert float_cmp_all(nabla_Js[idx].to_numpy(), model.gradient(us[idx], ps[idx], qs[idx], alpha=0).to_numpy(), REL_TOL, ABS_TOL)

@pytest.mark.parametrize("config", test_configs_lin_J, ids=[config['setup_name'] for config in test_configs_lin_J])
def test_lin_Js(config: Dict) -> None:
    qs = config['qs']
    ds = config['ds']
    us = config['us']
    lin_us = config['lin_us']
    lin_Js = config['lin_Js']
    alphas = config['alphas']
    model = config['model']

    for idx in range(len(qs)):
        assert float_cmp_all(lin_Js[idx], model.linearized_objective(qs[idx], ds[idx], us[idx], lin_us[idx], alphas[idx]), REL_TOL, ABS_TOL)

@pytest.mark.parametrize("config", test_configs_nabla_lin_J, ids=[config['setup_name'] for config in test_configs_nabla_lin_J])
def test_nabla_lin_Js(config: Dict) -> None:
    qs = config['qs']
    ds = config['ds']
    us = config['us']
    lin_ps = config['lin_ps']
    nabla_lin_Js = config['nabla_lin_Js']
    alphas = config['alphas']
    model = config['model']

    for idx in range(len(qs)):
        assert float_cmp_all(nabla_lin_Js[idx].to_numpy(), model.linearized_gradient(qs[idx], ds[idx], us[idx], lin_ps[idx], alphas[idx]).to_numpy(), REL_TOL, ABS_TOL)


if __name__ == '__main__':
    test_us()