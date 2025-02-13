import pytest
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import Union, Callable, Tuple, List, Dict

from pymor.basic import *

from RBInvParam.problems.problems import build_InstationaryModelIP
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.reductor import InstationaryModelIPReductor
from RBInvParam.model import InstationaryModelIP

# import os
    # os.remove('')

CWD = Path(__file__).parent.resolve()
spec = importlib.util.spec_from_file_location('setups',  CWD / './setups.py')
setups = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setups)
SETUPS = setups.SETUPS

set_log_levels({
    'pymor' : 'WARN'
})

ABS_TOL = 1e-14
REL_TOL = 1e-14

logger = get_default_logger(logger_name='test_gradient')

configs = []
for setup_name, setup in SETUPS.items():        
    FOM = build_InstationaryModelIP(setup, logger=logger)
    reductor = InstationaryModelIPReductor(FOM)

    q = FOM.Q.make_array(FOM.setup['model_parameter']['q_circ'])
    #q = FOM.Q.make_array(FOM.model_parameter['q_circ'])
    u = FOM.solve_state(q)
    p = FOM.solve_adjoint(q, u)
    J = FOM.objective(u)
    nabla_J = FOM.gradient(u, p, q)

    reductor.extend_basis(
        U = nabla_J,
        basis = 'parameter_basis'
    )
    QrFOM = reductor.reduce()


    state_shapshots = FOM.V.empty()
    # TODO HaPOD
    state_shapshots.append(u)
    state_shapshots.append(p)

    reductor.extend_basis(
        U = state_shapshots,
        basis = 'state_basis'
    )
    QrVrROM = reductor.reduce()

    configs.append({
        'model_name' : setup_name + '_FOM',
        'alpha' : 1e-1,
        'model' : FOM
    })

    # configs.append({
    #     'model_name' : setup_name + '_QrFOM',
    #     'alpha' : 1e0,
    #     'model' : QrFOM
    # })

    # configs.append({
    #     'model_name' : setup_name + '_QrVrROM',
    #     'alpha' : 1e0,
    #     'model' : QrVrROM
    # })


def derivative_check(model : InstationaryModelIP ,
                     f : Callable, 
                     df : Callable, 
                     save_path: Union[str, Path],
                     title: str) -> Tuple[List, List]:
    Eps = np.array([10**(-i) for i in range(0,13)])

    model_dims = model.setup['dims']
    if model.q_time_dep:
        q = model.Q.make_array(model.setup['model_parameter']['q_circ'])
        dq = model.Q.make_array(model.setup['model_parameter']['q_circ'])
    else:
        q = model.Q.make_array(model.setup['model_parameter']['q_circ'])
        dq = model.Q.make_array(model.setup['model_parameter']['q_circ'])

    T = np.zeros(np.shape(Eps))
    fq = f(q)
    df = df(q)

    if model.q_time_dep: 
        product = model.products['bochner_prod_Q']
    else:
        product = model.products['prod_Q']

    norm_dq = np.sqrt(product.apply2(dq, dq))[0,0]
    assert norm_dq > 0
    dq = 1 / norm_dq * dq
    
    # Compute central & right-side difference quotient
    for i in range(len(Eps)):
        h = Eps[i]*dq 
        f_plus = f(q+h)

        if model.riesz_rep_grad:
            if model.q_time_dep: 
                df_h = model.products['bochner_prod_Q'].apply2(df, h)[0,0]
                #df_h = np.sum(df.pairwise_inner(h, product=model.products['prod_Q']))
            else:
                df_h = model.products['prod_Q'].pairwise_apply2(df, h)
                #df_h = model.products['prod_Q'].pairwise_apply2(df, h)
        else:
            if model.q_time_dep: 
                df_h = np.sum(df.pairwise_inner(h))
                #df_h = np.sum(df.pairwise_inner(h))
            else:
                df_h = df.inner(h)[0,0]
                #df_h = df.inner(h)[0,0]

        T[i] = np.abs(f_plus - fq - df_h)

    plt.figure()
    plt.xlabel('$eps$')
    plt.ylabel('$J$')
    plt.loglog(Eps, Eps, label='O(eps)')
    plt.loglog(Eps, T, 'ro--',label='Test')
    plt.legend(loc='upper left')
    plt.grid()
    plt.title(title)
    plt.savefig(save_path)

    return (Eps, T)

#################################### FOM ####################################

# @pytest.mark.parametrize("config", configs, ids=[config['model_name'] for config in configs])
# def test_objective_gradient(config : Dict)-> None:
#     alpha = config['alpha']
#     model = config['model']
#     eps, diff_quot = derivative_check(
#         model,
#         model.compute_objective, 
#         model.compute_gradient,
#         Path('./test_gradient_dumps/' + config['model_name'] 
#              + '_' + sys._getframe().f_code.co_name + '.png'),
#         title = 'J'
#     )
#     assert np.all(eps > diff_quot)

# @pytest.mark.parametrize("config", configs, ids=[config['model_name'] for config in configs])
# def test_regularization_term_gradient(config : Dict) -> None:
#     alpha = config['alpha']
#     model = config['model']
#     eps, diff_quot = derivative_check(
#         model,
#         lambda q: alpha * model.regularization_term(q), 
#         lambda q: alpha * model.gradient_regularization_term(q),
#         Path('./test_gradient_dumps/' + config['model_name'] 
#              + '_' + sys._getframe().f_code.co_name + '.png'),
#         title = 'regularization'
#     )

#     assert np.all(eps > diff_quot)

@pytest.mark.parametrize("config", configs, ids=[config['model_name'] for config in configs])
def test_linearized_objective_gradient(config : Dict)-> None:
    alpha = config['alpha']
    model = config['model']
    q = model.Q.make_array(model.setup['model_parameter']['q_circ'])
    eps, diff_quot = derivative_check(
        model,
        lambda d : model.compute_linearized_objective(q, d, alpha),
        lambda d: model.compute_linearized_gradient(q, d, alpha, use_cached_operators=True),
        Path('./test_gradient_dumps/' + config['model_name'] 
             + '_' + sys._getframe().f_code.co_name + '.png'),
        title = 'linearized_J'
    )
    assert np.all(eps > diff_quot)

@pytest.mark.parametrize("config", configs, ids=[config['model_name'] for config in configs])
def test_linearized_regularization_term_gradient(config : Dict) -> None:
    alpha = config['alpha']
    model = config['model']

    q = model.Q.make_array(model.setup['model_parameter']['q_circ'])
    eps, diff_quot = derivative_check(
        model,
        lambda d: alpha * model.linearized_regularization_term(q, d), 
        lambda d: alpha * model.linarized_gradient_regularization_term(q, d),
        Path('./test_gradient_dumps/' + config['model_name'] 
             + '_' + sys._getframe().f_code.co_name + '.png'),
        title = 'linearized_regularization'
    )

    assert np.all(eps > diff_quot)