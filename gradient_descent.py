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
    
    delta_iterate = previous_iterate - pre_previous_iterate
    delta_gradient = previous_gradient - pre_previous_gradient
    # TODO What is the inner product on Q^K
    step_size = product.pairwise_apply2(delta_iterate, delta_gradient).sum() / product.pairwise_apply2(delta_gradient, delta_gradient).sum()
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
    inital_step_size: float,
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

    logger.info(f"Initial objective = {current_J}.")
    
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

        #grad.scal(1.0 / norm_grad)

        if i < 2:
        #if i > -1:
            grad.scal(1.0 / norm_grad)
            current_d, current_J = armijo_line_serach(
                previous_iterate = previous_d,
                previous_value = previous_J,
                search_direction = -grad,
                func = lambda d: model.compute_linearized_objective(q, d, alpha),
                inital_step_size = inital_step_size)       

            if (i % 10 == 0):
                logger.info(f"  Iteration {i+1} of {int(max_iter)} : objective = {current_J:3.4e}, norm gradient = {np.linalg.norm(model.compute_linearized_gradient(q, current_d, alpha).to_numpy()):3.4e}.")
                
        else:
            current_d, current_J = barzilai_borwein_line_serach(
                previous_iterate =  buffer_d[-1],
                pre_previous_iterate = buffer_d[-2],
                previous_gradient = buffer_nabla_J[-1],
                pre_previous_gradient = buffer_nabla_J[-2],
                product=model.products['prod_Q'],
                search_direction = grad,
                func = lambda d: model.compute_linearized_objective(q, d, alpha),

            )

            if (i % 10 == 0):
                logger.info(f"  Iteration {i+1} of {int(max_iter)} : objective = {current_J:3.4e}, norm gradient = {np.linalg.norm(model.compute_linearized_gradient(q, current_d, alpha).to_numpy()):3.4e}.")
                logger.info(f"  inital_step_size = {str(inital_step_size)}")

        buffer_d.pop(0)
        buffer_d.append(current_d)

        buffer_J.pop(0)
        buffer_J.append(current_J)    

        #stagnation check
        if i > 3:
            if abs(buffer_J[0] - buffer_J[1]) < MACHINE_EPS and abs(buffer_J[1] - buffer_J[2]) < MACHINE_EPS:
                logger.info(f"Stop at iteration {i+1} of {int(max_iter)}, due to stagnation.")
                break

    if converged:
        logger.info(f"Converged at iteration {last_i} of {int(max_iter)}.")
    else:
        logger.info(f"NOT converged after {int(max_iter)} iterations.")

    logger.info(f"objective = {current_J:3.4e}, norm gradient = {np.linalg.norm(model.compute_linearized_gradient(q, current_d, alpha).to_numpy()):3.4e}.")

    return current_d


# def gradient_descent_non_linearized_problem(
#     model : InstationaryModelIP,
#     q_start : np.array,
#     alpha : float,
#     max_iter : int,
#     tol : float,
#     inital_step_size: float,
#     logger: logging.Logger = None
# ) -> np.array:
#     assert alpha >= 0
#     assert tol > 0
#     assert inital_step_size > 0

#     if not logger:
#         logger = logging.getLogger('gradient_descent')
#         logger.setLevel(logging.DEBUG)

#     previous_q = np.nan
#     current_q = model.Q.make_array(q_start)
#     converged = False
#     last_i = -np.inf
#     buffer = [-np.inf, -np.inf, -np.inf]

#     previous_J = np.inf
#     current_J = model.compute_objective(current_q, alpha)
#     logger.info(f"Initial objective = {current_J}.")
    
#     for i in range(int(max_iter)):
#         previous_q = current_q.copy()
#         previous_J = current_J.copy()

#         grad = model.compute_gradient(previous_q, alpha)
#         if np.linalg.norm(grad.to_numpy()) < tol:
#             last_i = i
#             converged = True
#             break
        
#         grad.scal(1.0 / np.linalg.norm(grad.to_numpy()))

#         step_size = inital_step_size
#         current_q = previous_q - step_size * grad
#         current_J = model.compute_objective(current_q, alpha)
        
#         # TODO Barzilai-Bornwein
#         # if not armijo_condition(previous_J, current_J, step_size, previous_q, current_q, kappa_arm=1e-4):
#         #     #if inital_step_size > 1:
#         #     inital_step_size = 0.95 * inital_step_size
#         # else:
#         #     inital_step_size = 1.05 * inital_step_size

#         while not armijo_condition(previous_J, current_J, step_size, previous_q, current_q, kappa_arm=1e-4):
#             step_size = 0.5 * step_size
#             current_q = previous_q - step_size * grad
#             current_J = model.compute_objective(current_q, alpha)

#         if (i % 10 == 0):
#             logger.info(f"  Iteration {i+1} of {int(max_iter)} : objective = {current_J:3.4e}, norm gradient = {np.linalg.norm(model.compute_gradient(current_q, alpha).to_numpy()):3.4e}.")
#             logger.info(f"  inital_step_size = {str(inital_step_size)}")
        
#         buffer.pop(0)
#         buffer.append(current_J)
        
#         #stagnation check
#         if i > 3:
#             if abs(buffer[0] - buffer[1]) < MACHINE_EPS and abs(buffer[1] -buffer[2]) < MACHINE_EPS:
#                 logger.info(f"Stop at iteration {i+1} of {int(max_iter)}, due to stagnation.")
#                 break


#     if converged:
#         logger.info(f"Converged at iteration {last_i} of {int(max_iter)}.")
#     else:
#         logger.info(f"NOT converged after {int(max_iter)} iterations.")

#     logger.info(f"objective = {current_J:3.4e}, norm gradient = {np.linalg.norm(model.compute_gradient(current_q, alpha).to_numpy()):3.4e}.")

#     return current_q
