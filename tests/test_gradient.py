import pytest
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import Tuple, List

from pymor.basic import *

from RBInvParam.utils.io import load_FOM_from_config
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.reductor import InstationaryModelIPReductor


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

#TODO This is wrong for more than one config
#for config_name, config in CONFIGS.items():
config = CONFIGS['default_config_q_time_dep']
#config = CONFIGS['default_config_q_non_time_dep']

FOM = load_FOM_from_config(config, logger=logger)

reductor = InstationaryModelIPReductor(FOM)

q = FOM.Q.make_array(FOM.model_parameter['q_circ'])
u = FOM.solve_state(q)
p = FOM.solve_adjoint(q, u)
J = FOM.objective(u)
nabla_J = FOM.gradient(u, p)

reductor.extend_basis(
    U = nabla_J,
    basis = 'parameter_basis'
)
Qr_ROM = reductor.reduce()




def derivative_check(model,f, df, save_path, mode = 1) -> Tuple[List, List]:
    Eps = np.array([1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6])
    
    if model.model_parameter['q_time_dep']:
        q  = model.Q.make_array(np.random.random((model.dims['nt'], model.dims['par_dim'])))
        dq = model.Q.make_array(np.random.random((model.dims['nt'], model.dims['par_dim'])))
    else:
        q  = model.Q.make_array(np.random.random((1, model.dims['par_dim'])))
        dq = model.Q.make_array(np.random.random((1, model.dims['par_dim'])))

    T = np.zeros(np.shape(Eps))
    T2 = T
    fq = f(q)
    df = df(q)
    # TODO make consitent definition of the delta_t 
    product = model.products['bochner_prod_Q'] 
    norm_dq = np.sqrt(product.apply2(dq, dq))
    assert norm_dq > 0
    dq = 1 / (norm_dq / model.delta_t) * dq
    
    # Compute central & right-side difference quotient
    for i in range(len(Eps)):
        h = Eps[i]*dq 
        f_plus = f(q+h)
        df_h = product.apply2(df, h) / model.delta_t 
        T[i] = np.abs(f_plus - fq - df_h)

    #Plot
    plt.figure()
    plt.xlabel('$eps$')
    plt.ylabel('$J$')
    plt.loglog(Eps, Eps, label='O(eps)')
    plt.loglog(Eps, T, 'ro--',label='Test')
    plt.legend(loc='upper left')
    plt.grid()
    plt.title("Rightside difference quotient")
    plt.savefig(save_path)

    return (Eps, T2)

#################################### FOM ####################################

def test_FOM_objective_gradient()-> None:
    model = FOM
    eps, diff_quot = derivative_check(
        model,
        model.compute_objective, 
        model.compute_gradient,
        Path('./' + sys._getframe().f_code.co_name + '.png')
    )
    assert np.all(eps > diff_quot)


def test_FOM_regularization_term_gradient() -> None:
    alpha = 1e0
    model = FOM
    eps, diff_quot = derivative_check(
        model,
        lambda q: alpha * model.regularization_term(q), 
        lambda q: alpha * model.gradient_regularization_term(q),
        Path('./' + sys._getframe().f_code.co_name + '.png')
    )
    assert np.all(eps > diff_quot)


def test_FOM_linearized_objective_gradient()-> None:
    alpha = 1e0
    model = FOM
    q = model.Q.make_array(model.model_parameter['q_circ'])
    eps, diff_quot = derivative_check(
        model,
        lambda d : model.compute_linearized_objective(q, d, alpha),
        lambda d: model.compute_linearized_gradient(q, d, alpha),
        Path('./' + sys._getframe().f_code.co_name + '.png')
    )
    assert np.all(eps > diff_quot)

#################################### Qr-ROM ####################################

def test_Qr_ROM_objective_gradient()-> None:
    model = Qr_ROM
    eps, diff_quot = derivative_check(
        model,
        model.compute_objective, 
        model.compute_gradient,
        Path('./' + sys._getframe().f_code.co_name + '.png')
    )
    assert np.all(eps > diff_quot)


def test_Qr_ROM_regularization_term_gradient() -> None:
    alpha = 1e0
    model = Qr_ROM
    eps, diff_quot = derivative_check(
        model,
        lambda q: alpha * model.regularization_term(q), 
        lambda q: alpha * model.gradient_regularization_term(q),
        Path('./' + sys._getframe().f_code.co_name + '.png')
    )
    assert np.all(eps > diff_quot)


def test_Qr_ROM_linearized_objective_gradient()-> None:
    alpha = 1e0
    model = Qr_ROM
    q = model.Q.make_array(model.model_parameter['q_circ'])
    eps, diff_quot = derivative_check(
        model,
        lambda d : model.compute_linearized_objective(q, d, alpha),
        lambda d: model.compute_linearized_gradient(q, d, alpha),
        Path('./' + sys._getframe().f_code.co_name + '.png')
    )
    assert np.all(eps > diff_quot)









# q = FOM.Q.make_array(FOM.model_parameter['q_circ'])
# d = FOM.Q.make_array(q)
# q_r = reductor.project_vectorarray(q, 'parameter_basis')
# q_r = Q_r_ROM.Q.make_array(q_r)
# d_r = reductor.project_vectorarray(d, 'parameter_basis')
# d_r = Q_r_ROM.Q.make_array(d_r)



if __name__ == "__main__":
    test_FOM_objective_gradient()

    # q = FOM.Q.make_array(FOM.model_parameter['q_circ'])
    # print(q)

    # print(FOM.regularization_term(q))
    # print(FOM.gradient_regularization_term(q))

