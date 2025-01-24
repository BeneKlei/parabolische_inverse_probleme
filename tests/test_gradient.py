import pytest
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import Union, Callable, Tuple, List

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

logger = get_default_logger()

#TODO This is wrong for more than one setup
#setup = SETUPS['default_setup_q_time_dep']
setup = SETUPS['diffusion_setup_q_time_dep']

FOM = build_InstationaryModelIP(setup, logger=logger)

reductor = InstationaryModelIPReductor(FOM)

q = FOM.Q.make_array(FOM.setup['model_parameter']['q_circ'])
#q = FOM.Q.make_array(FOM.model_parameter['q_circ'])
u = FOM.solve_state(q)
p = FOM.solve_adjoint(q, u)
J = FOM.objective(u)
nabla_J = FOM.gradient(u, p)

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

def derivative_check(model : InstationaryModelIP ,
                     f : Callable, 
                     df : Callable, 
                     save_path: Union[str, Path]) -> Tuple[List, List]:
    Eps = np.array([10**(-i) for i in range(0,15)])

    model_dims = model.setup['dims']
    if model.q_time_dep:
        q  = model.Q.make_array(np.random.random((model_dims['nt'], model_dims['par_dim'])))
        dq = model.Q.make_array(np.random.random((model_dims['nt'], model_dims['par_dim'])))
    else:
        q  = model.Q.make_array(np.random.random((1, model_dims['par_dim'])))
        dq = model.Q.make_array(np.random.random((1, model_dims['par_dim'])))

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

        if model.q_time_dep: 
            df_h = np.sum(df.pairwise_inner(h))
        else:
            df_h = df.inner(h)[0,0]

        T[i] = np.abs(f_plus - fq - df_h)

    plt.figure()
    plt.xlabel('$eps$')
    plt.ylabel('$J$')
    plt.loglog(Eps, Eps, label='O(eps)')
    plt.loglog(Eps, T, 'ro--',label='Test')
    plt.legend(loc='upper left')
    plt.grid()
    plt.title("Rightside difference quotient")
    plt.savefig(save_path)

    return (Eps, T)

#################################### FOM ####################################

def test_FOM_objective_gradient()-> None:
    model = FOM
    eps, diff_quot = derivative_check(
        model,
        model.compute_objective, 
        model.compute_gradient,
        Path('./test_gradient_dumps/' + sys._getframe().f_code.co_name + '.png')
    )
    #assert np.all(eps > diff_quot)


def test_FOM_regularization_term_gradient() -> None:
    alpha = 1e0
    model = FOM
    eps, diff_quot = derivative_check(
        model,
        lambda q: alpha * model.regularization_term(q), 
        lambda q: alpha * model.gradient_regularization_term(q),
        Path('./test_gradient_dumps/' + sys._getframe().f_code.co_name + '.png')
    )

    assert np.all(eps > diff_quot)


def test_FOM_linearized_objective_gradient()-> None:
    alpha = 1e0
    model = FOM
    q = model.Q.make_array(model.setup['model_parameter']['q_circ'])
    eps, diff_quot = derivative_check(
        model,
        lambda d : model.compute_linearized_objective(q, d, alpha),
        lambda d: model.compute_linearized_gradient(q, d, alpha),
        Path('./test_gradient_dumps/' + sys._getframe().f_code.co_name + '.png')
    )
    assert np.all(eps > diff_quot)

#################################### Qr-FOM ####################################

def test_QrFOM_objective_gradient()-> None:
    model = QrFOM
    eps, diff_quot = derivative_check(
        model,
        model.compute_objective, 
        model.compute_gradient,
        Path('./test_gradient_dumps/' + sys._getframe().f_code.co_name + '.png')
    )
    assert np.all(eps > diff_quot)


def test_QrFOM_regularization_term_gradient() -> None:
    alpha = 1e0
    model = QrFOM
    eps, diff_quot = derivative_check(
        model,
        lambda q: alpha * model.regularization_term(q), 
        lambda q: alpha * model.gradient_regularization_term(q),
        Path('./test_gradient_dumps/' + sys._getframe().f_code.co_name + '.png')
    )
    assert np.all(eps > diff_quot)


def test_QrFOM_linearized_objective_gradient()-> None:
    alpha = 1e0
    model = QrFOM
    q = model.Q.make_array(model.setup['model_parameter']['q_circ'])
    eps, diff_quot = derivative_check(
        model,
        lambda d : model.compute_linearized_objective(q, d, alpha),
        lambda d: model.compute_linearized_gradient(q, d, alpha),
        Path('./test_gradient_dumps/' + sys._getframe().f_code.co_name + '.png')
    )
    assert np.all(eps > diff_quot)

#################################### Qr-VrROM ####################################

def test_QrVrROM_objective_gradient()-> None:
    model = QrVrROM
    eps, diff_quot = derivative_check(
        model,
        model.compute_objective, 
        model.compute_gradient,
        Path('./test_gradient_dumps/' + sys._getframe().f_code.co_name + '.png')
    )
    assert np.all(eps > diff_quot)


def test_QrVrROM_regularization_term_gradient() -> None:
    alpha = 1e0
    model = QrVrROM
    eps, diff_quot = derivative_check(
        model,
        lambda q: alpha * model.regularization_term(q), 
        lambda q: alpha * model.gradient_regularization_term(q),
        Path('./test_gradient_dumps/' + sys._getframe().f_code.co_name + '.png')
    )
    assert np.all(eps > diff_quot)


def test_QrVrROM_linearized_objective_gradient()-> None:
    alpha = 1e0
    model = QrVrROM
    q = model.Q.make_array(model.setup['model_parameter']['q_circ'])
    eps, diff_quot = derivative_check(
        model,
        lambda d : model.compute_linearized_objective(q, d, alpha),
        lambda d: model.compute_linearized_gradient(q, d, alpha),
        Path('./test_gradient_dumps/' + sys._getframe().f_code.co_name + '.png')
    )
    assert np.all(eps > diff_quot)



if __name__ == "__main__":
    test_FOM_regularization_term_gradient()
    #test_FOM_objective_gradient()
    #test_QrFOM_regularization_term_gradient()

    # q = FOM.Q.make_array(FOM.model_parameter['q_circ'])
    # print(q)

    # print(FOM.regularization_term(q))
    # print(FOM.gradient_regularization_term(q))

