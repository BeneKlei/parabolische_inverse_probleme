import pytest
import importlib.util
from pathlib import Path

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

for setup_name, setup in SETUPS.items():        
    _, FOM = build_InstationaryModelIP(setup, logger) 

    path = CWD / Path(f'./{setup_name}.pkl')
    assert path.exists()
    with open(path, 'rb') as file:
        solutions = load(file)

    qs = solutions['qs']
    ds = solutions['ds']
    alphas = solutions['alphas']
    us = solutions['us']
    ps = solutions['ps']
    lin_us = solutions['lin_us']
    lin_ps = solutions['lin_ps']
    Js = solutions['Js']
    nabla_Js = solutions['nabla_Js']
    lin_Js = solutions['lin_Js']
    nabla_lin_Js = solutions['nabla_lin_Js']

def test_us() -> None:
    for idx in range(len(qs)):
        assert float_cmp_all(us[idx].to_numpy(), FOM.solve_state(qs[idx]).to_numpy(), REL_TOL, ABS_TOL)

def test_ps() -> None:
    for idx in range(len(qs)):
        assert float_cmp_all(ps[idx].to_numpy(), FOM.solve_adjoint(qs[idx], us[idx]).to_numpy(), REL_TOL, ABS_TOL)

def test_lin_us() -> None:
    for idx in range(len(qs)):
        assert float_cmp_all(lin_us[idx].to_numpy(), FOM.solve_linearized_state(qs[idx], ds[idx], us[idx]).to_numpy(), REL_TOL, ABS_TOL)

def test_lin_ps() -> None:
    for idx in range(len(qs)):
        assert float_cmp_all(lin_ps[idx].to_numpy(), FOM.solve_linearized_adjoint(qs[idx], us[idx], lin_us[idx]).to_numpy(), REL_TOL, ABS_TOL)

def test_Js() -> None:
    for idx in range(len(qs)):
        assert float_cmp_all(Js[idx], FOM.objective(us[idx], qs[idx], alpha=0), REL_TOL, ABS_TOL)

def test_nabla_Js() -> None:
    for idx in range(len(qs)):
        assert float_cmp_all(nabla_Js[idx].to_numpy(), FOM.gradient(us[idx], ps[idx], qs[idx], alpha=0).to_numpy(), REL_TOL, ABS_TOL)

def test_lin_Js() -> None:
    for idx in range(len(qs)):
        assert float_cmp_all(lin_Js[idx], FOM.linearized_objective(qs[idx], ds[idx], us[idx], lin_us[idx], alphas[idx]), REL_TOL, ABS_TOL)

def test_nabla_lin_Js() -> None:
    for idx in range(len(qs)):
        assert float_cmp_all(nabla_lin_Js[idx].to_numpy(), FOM.linearized_gradient(qs[idx], ds[idx], us[idx], lin_ps[idx], alphas[idx]).to_numpy(), REL_TOL, ABS_TOL)

