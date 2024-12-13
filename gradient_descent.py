import numpy as np
import logging

from model import InstationaryModelIP

MACHINE_EPS = 1e-16

def armijo_condition(
    previous_J : float,
    current_J : float,
    step_size : float,
    previous : np.array,
    current : np.array,
    kappa_arm: float) -> bool:
     
    norm_d = np.linalg.norm(previous.to_numpy() - current.to_numpy())
    lhs =  previous_J - current_J
    rhs = kappa_arm / step_size * norm_d**2

    if abs(lhs) <= MACHINE_EPS:
        lhs = 0

    if abs(rhs) <= MACHINE_EPS:
        rhs = 0

    return lhs >= rhs

    

def gradient_descent_linearized_problem(
    model : InstationaryModelIP,
    q : np.array,
    d_start : np.array,
    alpha : float,
    max_iter : int,
    tol : float,
    inital_step_size: float,
    logger: logging.Logger = None,
) -> np.array:
    assert alpha >= 0
    assert tol > 0
    assert inital_step_size > 0

    if not logger:
        logger = logging.getLogger('gradient_descent')
        logger.setLevel(logging.DEBUG)

    previous_d = np.nan
    current_d = d_start
    converged = False
    last_i = -np.inf
    buffer = [-np.inf, -np.inf, -np.inf]

    previous_J = np.inf
    current_J = model.compute_linearized_objective(q, current_d, alpha)
    logger.info(f"Initial objective = {current_J}.")
    
    for i in range(int(max_iter)):
        previous_d = current_d.copy()
        previous_J = current_J.copy()

        grad = model.compute_linearized_gradient(q, previous_d, alpha)
        if np.linalg.norm(grad.to_numpy()) < tol:
            last_i = i
            converged = True
            break

        step_size = inital_step_size
        current_d = previous_d - step_size * grad
        current_J = model.compute_linearized_objective(q, current_d, alpha)
        
        # TODO Barzilai-Bornwein
        if not armijo_condition(previous_J, current_J, step_size, previous_d, current_d, kappa_arm=1e-4):
            inital_step_size = 0.5 * inital_step_size
        else:
            inital_step_size = 1.05 * inital_step_size
            

        while not armijo_condition(previous_J, current_J, step_size, previous_d, current_d, kappa_arm=1e-4):
            step_size = 0.5 * step_size
            current_d = previous_d - step_size * grad
            current_J = model.compute_linearized_objective(q, current_d, alpha)

        if (i % 10 == 0):
            logger.info(f"  Iteration {i+1} of {int(max_iter)} : objective = {current_J}, norm gradient = {np.linalg.norm(model.compute_linearized_gradient(q, current_d, alpha).to_numpy())}.")
            logger.info(f"  inital_step_size = {str(inital_step_size)}")

        buffer.pop(0)
        buffer.append(current_J)
        #stagnation check
        if i > 3:
            # print(buffer)
            # print(abs(buffer[0] - buffer[1]))
            # print(abs(buffer[1] - buffer[2]))
            if abs(buffer[0] - buffer[1]) < MACHINE_EPS and abs(buffer[1] -buffer[2]) < MACHINE_EPS:
                logger.info(f"Stop at iteration {i+1} of {int(max_iter)}, due to stagnation.")
                break
            



    if converged:
        logger.info(f"Converged at iteration {last_i} of {int(max_iter)}.")
    else:
        logger.info(f"NOT converged after {int(max_iter)} iterations.")

    # print(current_J)
    # print(q)
    # print(current_d)
    # print(model.compute_linearized_objective(q, current_d, alpha))
    logger.info(f"objective = {current_J}, norm gradient = {np.linalg.norm(model.compute_linearized_gradient(q, current_d, alpha).to_numpy())}.")

    return current_d


def gradient_descent_non_linearized_problem(
    model : InstationaryModelIP,
    q_start : np.array,
    alpha : float,
    max_iter : int,
    tol : float,
    inital_step_size: float,
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
    converged = False
    last_i = -np.inf

    previous_J = np.inf
    current_J = model.compute_objective(current_q, alpha)
    logger.info(f"Initial objective = {current_J}.")
    
    for i in range(int(max_iter)):
        previous_q = current_q.copy()
        previous_J = current_J.copy()

        grad = model.compute_gradient(previous_q, alpha)
        if np.linalg.norm(grad.to_numpy()) < tol:
            last_i = i
            converged = True
            break

        step_size = inital_step_size
        current_q = previous_q - step_size * grad
        current_J = model.compute_objective(current_q, alpha)
        
        # TODO Barzilai-Bornwein
        if not armijo_condition(previous_J, current_J, step_size, previous_q, current_q, kappa_arm=1e-4):
            #if inital_step_size > 1:
            inital_step_size = 0.95 * inital_step_size
        else:
            inital_step_size = 1.05 * inital_step_size

        while not armijo_condition(previous_J, current_J, step_size, previous_q, current_q, kappa_arm=1e-4):
            step_size = 0.5 * step_size
            current_q = previous_q - step_size * grad
            current_J = model.compute_objective(current_q, alpha)

        if (i % 1 == 0):
            logger.info(f"  Iteration {i+1} of {int(max_iter)} : objective = {current_J}, norm gradient = {np.linalg.norm(model.compute_gradient(current_q, alpha).to_numpy())}.")
            logger.info(f"  inital_step_size = {str(inital_step_size)}")


    if converged:
        logger.info(f"Converged at iteration {last_i} of {int(max_iter)}.")
    else:
        logger.info(f"NOT converged after {int(max_iter)} iterations.")

    logger.info(f"objective = {current_J}, norm gradient = {np.linalg.norm(model.compute_gradient(current_q, alpha).to_numpy())}.")

    return current_q
