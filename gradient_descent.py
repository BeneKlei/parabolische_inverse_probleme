import numpy as np
import logging
from typing import Callable, Tuple

from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.operators.numpy import NumpyMatrixOperator

from model import InstationaryModelIP


MACHINE_EPS = 1e-16

def armijo_condition(
    previous_J : float,
    current_J : float,
    step_size : float,
    previous : NumpyVectorArray,
    current : NumpyVectorArray,
    kappa_arm: float = 1e-12) -> bool:
     
    norm_d = np.linalg.norm((previous - current).to_numpy())
    lhs =  previous_J - current_J
    rhs = kappa_arm / step_size * norm_d**2
    

    if abs(lhs) <= MACHINE_EPS:
        lhs = 0

    if abs(rhs) <= MACHINE_EPS:
        rhs = 0

    return lhs >= rhs

def armijo_line_serach(
        previous_iterate: NumpyVectorArray,
        previous_value: float,
        search_direction : NumpyVectorArray,
        func: Callable,
        inital_step_size: float) -> Tuple[NumpyVectorArray, float]:

        step_size = inital_step_size
        current_iterate = previous_iterate + step_size * search_direction
        current_value = func(current_iterate)

        while not armijo_condition(previous_value, current_value, step_size, previous_iterate, current_iterate, kappa_arm=1e-12):
            step_size = 0.5 * step_size
            current_iterate = previous_iterate + step_size * search_direction
            current_value = func(current_iterate)

        return (current_iterate, current_value)


def barzilai_borwein_line_serach(previous_iterate: NumpyVectorArray,
                                 pre_previous_iterate: NumpyVectorArray,
                                 previous_gradient: NumpyVectorArray,
                                 pre_previous_gradient: NumpyVectorArray,
                                 product : NumpyMatrixOperator,
                                 search_direction : NumpyVectorArray,
                                 func: Callable) -> Tuple[NumpyVectorArray, float]:
    
    # Using algorithm from
    # https://watermark.silverchair.com/8-1-141.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA2kwggNlBgkqhkiG9w0BBwagggNWMIIDUgIBADCCA0sGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM4WESSjzDIsa2gVMsAgEQgIIDHHeiS0AQ9L_1TmZFA7rg1QFF7cezo4BC_1OnBjkdHDIidNO_gGqaOBss8MNgz6cK5xd1mDqhTB0w0Jx202D40CtChSI6QCQUfSpoFcR3D28U6jRYZnaH9NjLIh0rE59cktmXZbX0aCov-NgLpmyfyrWVhK0hdkl5aXU_2hrh3b83hg2wjA_k9JVXwxHDhaS58iAtIv8Ulw4jBc8E6iV447KcH4RKuUT8PISqwQoTWF5-5564fSGEYVWrV2SFDbiHQpqnJBSLTJdXMK5EwqxXN3Z7b7byHqUe76bZdk5f2RROTuMX2TRITeGRZdnyqQ2qL2O2lmqqOrjomiKg3qUYcX_2wqBOyD2WC3cIjHalwNEgPZfRVAqJ-UCrBsnBdcwIPDlATYhN3XG-zMBUKURfQt8ypcoPlYQoZD0NI-d2Hsr7-Bx7ishcO8tJ07hY9tETD_KGtmZQCyAHpP5IqlRK00yo2XMvZc-_mhjc-f1UWrY9OGwGh5vaBaP7xvsmZnU60Pp-A4eKoqjwucTx3mv9PzhkZR-ZzqeuEfSMBP082-Hxh7WuLOk_YuRMvHbKEzkzVU9-9h9kMeZWxfYFMVgAoyE1nd3o1gTjYupKsS1LOAdMJKe-6r75K4ceV_C4aUocXxLbPQ8j154lil5ujc0ejPvW709tWQINj7SdvnSb5zydKyGIsT-3eMGMthuWsNoCEKZB6JHnTYkeDsNzbvfwUqSexaH-eJM5MiFDPVeft-OG-OQrQxc8xrVXEOF3sGjLCgtbpmkvgcDDNYTKIByb9d0O0Mzmznz6HzWNPxhHRH8ZEvQEMPCRbxQn0UQoq-9UcHUVwoZUXl_w9kZ9zUgNVK8kNQKSikmKaWhAfI_NGz5zzqP_4-G8FfG1Vbdwt2l0g3okGtEUI7IU2hofXY4ypnIlmv_7dsjqkLsxhWChcO9BIdAyQ0svinQRW22b0ClgAQejDhpkJDwm8PJY03JKPgX9223GJ2zjXRO9Fx-OZV0TwWgt1xwhMbRenK6aJHDfnhsGZPQ5JjWAkMP_DQk8ZAGefCOHiYKdZZjpah0

    # print(previous_iterate)
    # print(pre_previous_iterate)
    
    delta_iterate = previous_iterate - pre_previous_iterate
    delta_gradient = previous_gradient - pre_previous_gradient

    # TODO What is the inner product on Q^K

    #step_size = product(delta_iterate, delta_iterate) / product(delta_iterate, delta_gradient)
    step_size = product(delta_iterate, delta_gradient) / product(delta_gradient, delta_gradient)

    current_iterate = previous_iterate - step_size * search_direction
    current_value = func(current_iterate)

    return (current_iterate, current_value)



def gradient_descent_linearized_problem(
    model : InstationaryModelIP,
    q : np.array,
    d_start : np.array,
    alpha : float,
    max_iter : int,
    tol : float,
    inital_step_size: float = 1,
    logger: logging.Logger = None) -> np.array:
    assert alpha >= 0
    assert tol > 0
    assert inital_step_size > 0

    if not logger:
        logger = logging.getLogger('gradient_descent')
        logger.setLevel(logging.DEBUG)

    previous_d = np.nan
    current_d = d_start

    previous_J = np.inf
    current_J = model.compute_linearized_objective(q, current_d, alpha)

    converged = False
    last_i = -np.inf
    
    buffer_size = 3
    buffer_d = [np.nan for _ in range(buffer_size)]
    buffer_J = [np.inf for _ in range(buffer_size)]
    buffer_nabla_J = [np.nan for _ in range(buffer_size)]

    buffer_d.pop(0)
    buffer_d.append(current_d)

    buffer_J.pop(0)
    buffer_J.append(current_J)

    logger.info(f"Initial objective = {current_J:3.4e}.")
    
    for i in range(int(max_iter)):
        previous_d = current_d.copy()
        previous_J = current_J.copy()

        grad = model.compute_linearized_gradient(q, previous_d, alpha)
        buffer_nabla_J.pop(0)
        buffer_nabla_J.append(grad.copy())

        norm_grad = np.linalg.norm(grad.to_numpy())

        if norm_grad < tol:
            last_i = i
            converged = True
            break

        if i < 2:
        #if i > -1:
            grad.scal(1.0 / norm_grad)
            current_d, current_J = armijo_line_serach(
                previous_iterate = previous_d,
                previous_value = previous_J,
                search_direction = -grad,
                func = lambda d: model.compute_linearized_objective(q, d, alpha),
                inital_step_size = inital_step_size)       

        else:
            current_d, current_J = barzilai_borwein_line_serach(
                previous_iterate =  buffer_d[-1],
                pre_previous_iterate = buffer_d[-2],
                previous_gradient = buffer_nabla_J[-1],
                pre_previous_gradient = buffer_nabla_J[-2],
                product=model.products['bochner_prod_Q'],
                search_direction = grad,
                func = lambda d: model.compute_linearized_objective(q, d, alpha))

        if (i % 10 == 0):
            logger.info(f"  Iteration {i+1} of {int(max_iter)} : objective = {current_J:3.4e}, norm gradient = {np.linalg.norm(model.compute_linearized_gradient(q, current_d, alpha).to_numpy()):3.4e}.")

        buffer_d.pop(0)
        buffer_d.append(current_d)

        buffer_J.pop(0)
        buffer_J.append(current_J)    

        #stagnation check
        if i > 10:
            if abs(buffer_J[0] - buffer_J[1]) < MACHINE_EPS and abs(buffer_J[1] - buffer_J[2]) < MACHINE_EPS:
                logger.info(f"Stop at iteration {i+1} of {int(max_iter)}, due to stagnation.")
                break

    if converged:
        logger.info(f"Gradient decent converged at iteration {last_i} of {int(max_iter)}.")
    else:
        logger.info(f"Gradient decent NOT converged after {int(max_iter)} iterations.")

    logger.info(f"objective = {current_J:3.4e}, norm gradient = {np.linalg.norm(model.compute_linearized_gradient(q, current_d, alpha).to_numpy()):3.4e}.")

    return current_d


def gradient_descent_non_linearized_problem(
    model : InstationaryModelIP,
    q_start : np.array,
    alpha : float,
    max_iter : int,
    tol : float,
    inital_step_size: float = 1,
    logger: logging.Logger = None
) -> np.array:
    assert alpha >= 0
    assert tol > 0
    assert inital_step_size > 0

    if not logger:
        logger = logging.getLogger('gradient_descent')
        logger.setLevel(logging.DEBUG)

    previous_q = np.nan
    current_q = model.Q.make_array(q_start)

    previous_J = np.inf
    current_J = model.compute_objective(current_q, alpha)

    converged = False
    last_i = -np.inf
    
    buffer_size = 3
    buffer_q = [np.nan for _ in range(buffer_size)]
    buffer_J = [np.inf for _ in range(buffer_size)]
    buffer_nabla_J = [np.nan for _ in range(buffer_size)]

    buffer_q.pop(0)
    buffer_q.append(current_q)

    buffer_J.pop(0)
    buffer_J.append(current_J)

    logger.info(f"Initial objective = {current_J}.")
    
    for i in range(int(max_iter)):
        previous_q = current_q.copy()
        previous_J = current_J.copy()

        grad = model.compute_gradient(previous_q, alpha)
        buffer_nabla_J.pop(0)
        buffer_nabla_J.append(grad.copy())

        norm_grad = np.linalg.norm(grad.to_numpy())
        
        if norm_grad < tol:
            last_i = i
            converged = True
            break
        
        if i < 2:
        #if i > -1:
            grad.scal(1.0 / norm_grad)
            current_q, current_J = armijo_line_serach(
                previous_iterate = previous_q,
                previous_value = previous_J,
                search_direction = -grad,
                func = lambda q: model.compute_objective(q, alpha),
                inital_step_size = inital_step_size)       

        else:
            current_q, current_J = barzilai_borwein_line_serach(
                previous_iterate =  buffer_q[-1],
                pre_previous_iterate = buffer_q[-2],
                previous_gradient = buffer_nabla_J[-1],
                pre_previous_gradient = buffer_nabla_J[-2],
                product=model.products['bochner_prod_Q'],
                search_direction = grad,
                func = lambda q: model.compute_objective(q, alpha))


        if (i % 10 == 0):
            logger.info(f"  Iteration {i+1} of {int(max_iter)} : objective = {current_J:3.4e}, norm gradient = {np.linalg.norm(model.compute_gradient(current_q, alpha).to_numpy()):3.4e}.")
        
        buffer_q.pop(0)
        buffer_q.append(current_q)

        buffer_J.pop(0)
        buffer_J.append(current_J)
        
        #stagnation check
        if i > 10:
            if abs(buffer_J[0] - buffer_J[1]) < MACHINE_EPS and abs(buffer_J[1] - buffer_J[2]) < MACHINE_EPS:
                logger.info(f"Stop at iteration {i+1} of {int(max_iter)}, due to stagnation.")
                break


    if converged:
        logger.info(f"Converged at iteration {last_i} of {int(max_iter)}.")
    else:
        logger.info(f"NOT converged after {int(max_iter)} iterations.")

    logger.info(f"objective = {current_J:3.4e}, norm gradient = {np.linalg.norm(model.compute_gradient(current_q, alpha).to_numpy()):3.4e}.")

    return current_q
