import numpy as np
import logging

from model import InstationaryModelIP

MACHINE_EPS = 1e-16

def armijo_condition(
    previous_J : float,
    current_J : float,
    step_size : float,
    previous_d : np.array,
    current_d : np.array,
    kappa_arm: float) -> bool:
     
    norm_d = np.linalg.norm(previous_d - current_d)
    lhs =  previous_J - current_J
    rhs = kappa_arm / step_size * norm_d**2

    if abs(lhs) <= MACHINE_EPS:
        lhs = 0

    if abs(rhs) <= MACHINE_EPS:
        rhs = 0

    return lhs >= rhs

    

def gradient_descent(
    model : InstationaryModelIP,
    q : np.array,
    d_start : np.array,
    alpha : float,
    max_iter : int,
    tol : float,
    inital_step_size: float
) -> np.array:
    assert alpha >= 0
    assert tol > 0
    assert inital_step_size > 0

    logger = logging.getLogger('gradient_descent')
    logger.setLevel(logging.INFO)

    previous_d = np.nan
    current_d = d_start

    previous_J = np.inf
    current_J = model.compute_linearized_objective(q, current_d, alpha)
    logger.info(f"Initial objective = {current_J}.")
    
    for i in range(int(max_iter)):
        previous_d = current_d.copy()
        previous_J = current_J.copy()

        grad = model.compute_linearized_gradient(q, previous_d, alpha).to_numpy()
        if np.linalg.norm(grad) < tol:
            break
            
        step_size = inital_step_size
        current_d = previous_d - step_size * grad
        current_J = model.compute_linearized_objective(q, current_d, alpha)
        
        if not armijo_condition(previous_J, current_J, step_size, previous_d, current_d, kappa_arm=1e-4):
            inital_step_size = 0.5 * inital_step_size

        while not armijo_condition(previous_J, current_J, step_size, previous_d, current_d, kappa_arm=1e-4):
            step_size = 0.5 * step_size
            current_d = previous_d - step_size * grad
            current_J = model.compute_linearized_objective(q, current_d, alpha)

            if (i % 1 == 0):
                logger.info('\t' + '-----------------------------------------')
                logger.info('\t' + str(step_size))

        if (i % 1 == 0):
            logger.info(f"Iteration {i+1}: objective = {current_J}, norm gradient = {np.linalg.norm(model.compute_linearized_gradient(q, current_d, alpha).to_numpy())}.")

    return current_d