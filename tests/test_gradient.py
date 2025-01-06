import pytest
import importlib.util
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, List

from pymor.basic import *
from pymor.tools.floatcmp import float_cmp_all
from pymor.core.pickle import load

from RBInvParam.utils.io import load_FOM_from_config
from RBInvParam.utils.logger import get_default_logger

# import os
    # os.remove('')

CWD = Path(__file__).parent.resolve()
spec = importlib.util.spec_from_file_location('configs',  CWD / './configs.py')
configs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configs)
CONFIGS = configs.CONFIGS

set_log_levels({
    'pymor' : 'WARN'
})

ABS_TOL = 1e-14
REL_TOL = 1e-14

logger = get_default_logger()

for config_name, config in CONFIGS.items():        
    FOM = load_FOM_from_config(config, logger=logger)

def derivative_check(model,f, df, save_path, mode = 1) -> Tuple[List, List]:
    
    print('derivative check ...')
    
    Eps = np.array([1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6])
    
    if model.model_parameter['q_time_dep']:
        q  = model.Q.make_array(np.random.random((model.dims['nt'], model.dims['par_dim'])))
        dq = model.Q.make_array(np.random.random((model.dims['nt'], model.dims['par_dim'])))
    else:
        q  = model.Q.make_array(np.random.random((1, model.dims['par_dim'])))
        dq = model.Q.make_array(np.random.random((1, model.dims['par_dim'])))

    T = np.zeros(np.shape(Eps))
    T2 = T
    ff = f(q)
    
    # Compute central & right-side difference quotient
    for i in range(len(Eps)):
        #print(Eps[i])
        f_plus = f(q+Eps[i]*dq)
        f_minus = f(q-Eps[i]*dq)
        
        dfq_np = df(q).to_numpy().T
        dq_np = dq.to_numpy().T
        df_dq = model.delta_t * np.sum(np.sum(dfq_np*dq_np, axis = 0))
        
        T[i] = abs( ( (f_plus - f_minus)/(2*Eps[i]) ) - df_dq )
        T2[i] =  abs( ( (f_plus - ff)/(Eps[i]) ) - df_dq )
        
    #Plot
    plt.figure()
    plt.xlabel('$eps$')
    plt.ylabel('$J$')
    plt.loglog(Eps, Eps, label='O(eps)')
    plt.loglog(Eps, T2, 'ro--',label='Test')
    plt.legend(loc='upper left')
    plt.grid()
    plt.title("Rightside difference quotient")
    plt.savefig(save_path)

    return (Eps, T2)

def test_objective_gradient()-> None:
    eps, diff_quot = derivative_check(
        FOM,
        FOM.compute_objective, 
        FOM.compute_gradient,
        './test_objective_gradient.png'
    )
    assert np.all(eps > diff_quot)


def test_regularization_term_gradient() -> None:
    alpha = 1e0
    eps, diff_quot = derivative_check(
        FOM,
        lambda q: alpha * FOM.regularization_term(q), 
        lambda q: alpha * FOM.gradient_regularization_term(q),
        './test_regularization_term_gradient.png'
    )
    assert np.all(eps > diff_quot)


def test_linearized_objective_gradient()-> None:
    alpha = 1e0
    q = FOM.Q.make_array(FOM.model_parameter['q_circ'])
    eps, diff_quot = derivative_check(
        FOM,
        lambda d : FOM.compute_linearized_objective(q, d, alpha),
        lambda d: FOM.compute_linearized_gradient(q, d, alpha),
        './test_linearized_objective_gradient.png'
    )
    assert np.all(eps > diff_quot)


if __name__ == "__main__":
    test_regularization_term_gradient()

