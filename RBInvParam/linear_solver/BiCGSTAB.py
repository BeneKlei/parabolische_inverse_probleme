import logging
import numpy as np
import scipy.sparse.linalg as spla
from typing import Dict, Tuple

from pymor.vectorarrays.interface import VectorArray

from RBInvParam.model import InstationaryModelIP

class IterationCounter:
    def __init__(self):
        self.count = 0
    def __call__(self, xk):
        self.count += 1

class NonTimeDepLinearGradientScipyOperator(spla.LinearOperator):
    def __init__(self,
                 model: InstationaryModelIP,
                 q: VectorArray,
                 alpha : float,
                 use_cached_operators: bool):

        assert isinstance(model, InstationaryModelIP)
        assert isinstance(q, VectorArray)
        assert q in model.Q
        assert alpha > 0
        assert len(q) == 1

        self.model = model
        self.q = q
        self.alpha = alpha
        self.use_cached_operators = use_cached_operators

        self.shape = (
            self.model.Q.dim, 
            self.model.Q.dim
        )
        self.dtype = np.dtype(np.float64)    
        
        self.b = model.compute_linearized_gradient(
            q, 
            model.Q.make_array(np.zeros(
                shape = (self.model.Q.dim)
            )), 
            alpha, 
            use_cached_operators=use_cached_operators
        ).to_numpy()[0]


    def _matvec(self, x: np.ndarray) -> None:
        return (self.model.compute_linearized_gradient(
            q=self.q, 
            d=self.model.Q.make_array(x), 
            alpha=self.alpha, 
            use_cached_operators=self.use_cached_operators
        ).to_numpy()[0] - self.b)

class TimeDepLinearGradientScipyOperator(spla.LinearOperator):
    def __init__(self,
            model: InstationaryModelIP,
            q: VectorArray,
            alpha : float,
            use_cached_operators: bool):

        assert isinstance(model, InstationaryModelIP)
        assert isinstance(q, VectorArray)
        assert q in model.Q
        assert alpha > 0
        assert len(q) == model.nt

        self.model = model
        self.q = q
        self.alpha = alpha
        self.use_cached_operators = use_cached_operators

        self.shape = (
            self.model.nt * self.model.Q.dim, 
            self.model.nt * self.model.Q.dim
        )
        self.dtype = np.dtype(np.float64)    
        
        self.b = model.compute_linearized_gradient(
            q, 
            model.Q.make_array(np.zeros(
                shape = (self.model.nt, self.model.Q.dim)
            )), 
            alpha, 
            use_cached_operators=use_cached_operators
        ).to_numpy().flatten()

    def _matvec(self, x : np.ndarray) -> None:
        return (self.model.compute_linearized_gradient(
            q=self.q, 
            d=self.model.Q.make_array(x.reshape((
                self.model.nt, self.model.Q.dim
            ))), 
            alpha=self.alpha, 
            use_cached_operators=self.use_cached_operators
        ).to_numpy().flatten() - self.b)

def BiCGStab_linearized_problem(
    model : InstationaryModelIP,
    q : VectorArray,
    d_start : VectorArray,
    alpha : float,
    lin_solver_parms : Dict,
    logger: logging.Logger,
    use_cached_operators: bool = False) -> Tuple[VectorArray, int]:

    counter = IterationCounter()

    if model.setup['model_parameter']['q_time_dep']:
        linear_solver_operator = TimeDepLinearGradientScipyOperator(
            model = model,
            q = q,
            alpha = alpha,
            use_cached_operators = use_cached_operators,
        )
        x0 = d_start.to_numpy().flatten()
    else:
        linear_solver_operator = NonTimeDepLinearGradientScipyOperator(
            model = model,
            q = q,
            alpha = alpha,
            use_cached_operators = use_cached_operators,
        )
        x0 = d_start.to_numpy()[0]
                        
    d, linear_solver_info = spla.bicgstab(
        A = linear_solver_operator,
        b = -linear_solver_operator.b,
        x0 = x0,
        rtol = lin_solver_parms['rtol'],
        atol = lin_solver_parms['atol'],
        maxiter = int(lin_solver_parms['maxiter']),
        callback=counter
    )
    
    if model.setup['model_parameter']['q_time_dep']:
        d = d.reshape((
            model.nt, model.Q.dim
        ))
    d = model.Q.make_array(d)

    maxiter = int(lin_solver_parms['maxiter'])    
    current_J = model.compute_linearized_objective(q, 
                                                   d, 
                                                   alpha, 
                                                   use_cached_operators=use_cached_operators)

    grad = model.compute_linearized_gradient(q, 
                                             d, 
                                             alpha, 
                                             use_cached_operators=use_cached_operators)

    norm_grad = model.compute_gradient_norm(grad)

    if counter.count < maxiter:
        logger.info(f"BiCGStab converged at iteration {counter.count} of {maxiter}.")
    else:
        logger.info(f"BiCGStab NOT converged after {maxiter} iterations.")

    logger.info(f"objective = {current_J:3.4e}, norm gradient = {norm_grad:3.4e}.")

    return d, counter.count