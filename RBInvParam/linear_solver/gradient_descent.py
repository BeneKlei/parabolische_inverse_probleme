import numpy as np
import logging
from typing import Callable, Tuple, Dict
from functools import partial

from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.vectorarrays.interface import VectorArray
from pymor.operators.numpy import NumpyMatrixOperator

from RBInvParam.model import InstationaryModelIP
from RBInvParam.domain_projector import SimpleBoundDomainProjector


MACHINE_EPS = 1e-16
CONV_TOL = 1e-16
#CONV_TOL = 1e-3

def armijo_condition(
    previous_J : float,
    current_J : float,
    step_size : float,
    previous : NumpyVectorArray,
    current : NumpyVectorArray,
    product : NumpyMatrixOperator,
    kappa_arm: float = 1e-12) -> bool:

    norm_d = product.apply2(previous - current,previous - current)[0,0]
    lhs =  previous_J - current_J
    #rhs = kappa_arm / step_size * norm_d**2
    rhs = kappa_arm / step_size * norm_d
    

    if abs(lhs) <= MACHINE_EPS:
        lhs = 0

    if abs(rhs) <= MACHINE_EPS:
        rhs = 0

    return lhs >= rhs

def armijo_line_serach(previous_iterate: NumpyVectorArray,
                       previous_value: float,
                       search_direction : NumpyVectorArray,
                       func: Callable,
                       product : NumpyMatrixOperator,
                       inital_step_size: float,
                       projector: SimpleBoundDomainProjector = None,
                       q: NumpyVectorArray = None,
                       kappa_arm: float = 1e-12,
                       min_step_size: float = 1e-20) -> Tuple[NumpyVectorArray, float, bool]:
        
        assert kappa_arm > 0
        assert min_step_size > 0

        stagnation_flag = False
        step_size = inital_step_size
        current_iterate = previous_iterate + step_size * search_direction
        
        if projector:
            current_iterate = projector.project_domain(q, current_iterate) - q

        current_value = func(current_iterate)
        
        condition = armijo_condition(previous_value, 
                                     current_value, 
                                     step_size, 
                                     previous_iterate, 
                                     current_iterate,
                                     product=product,
                                     kappa_arm=kappa_arm)
        


        while not condition:
            step_size = 0.5 * step_size
            if step_size < min_step_size:
                stagnation_flag = True
                break

            current_iterate = previous_iterate + step_size * search_direction

            if projector: 
                current_iterate = projector.project_domain(q, current_iterate) - q                

            current_value = func(current_iterate)
            
            condition = armijo_condition(previous_value, 
                                         current_value, 
                                         step_size, 
                                         previous_iterate, 
                                         current_iterate,
                                         product=product,
                                         kappa_arm=kappa_arm)

          
            

        return (current_iterate, current_value, stagnation_flag)

def barzilai_borwein_line_serach(previous_iterate: NumpyVectorArray,
                                 pre_previous_iterate: NumpyVectorArray,
                                 previous_gradient: NumpyVectorArray,
                                 pre_previous_gradient: NumpyVectorArray,
                                 product : NumpyMatrixOperator,
                                 search_direction : NumpyVectorArray,
                                 func: Callable, 
                                 projector: SimpleBoundDomainProjector = None,
                                 q: NumpyVectorArray = None,
                                 idx: int = 0) -> Tuple[NumpyVectorArray, float]:
    
    # Using algorithm from
    # https://watermark.silverchair.com/8-1-141.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA2kwggNlBgkqhkiG9w0BBwagggNWMIIDUgIBADCCA0sGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM4WESSjzDIsa2gVMsAgEQgIIDHHeiS0AQ9L_1TmZFA7rg1QFF7cezo4BC_1OnBjkdHDIidNO_gGqaOBss8MNgz6cK5xd1mDqhTB0w0Jx202D40CtChSI6QCQUfSpoFcR3D28U6jRYZnaH9NjLIh0rE59cktmXZbX0aCov-NgLpmyfyrWVhK0hdkl5aXU_2hrh3b83hg2wjA_k9JVXwxHDhaS58iAtIv8Ulw4jBc8E6iV447KcH4RKuUT8PISqwQoTWF5-5564fSGEYVWrV2SFDbiHQpqnJBSLTJdXMK5EwqxXN3Z7b7byHqUe76bZdk5f2RROTuMX2TRITeGRZdnyqQ2qL2O2lmqqOrjomiKg3qUYcX_2wqBOyD2WC3cIjHalwNEgPZfRVAqJ-UCrBsnBdcwIPDlATYhN3XG-zMBUKURfQt8ypcoPlYQoZD0NI-d2Hsr7-Bx7ishcO8tJ07hY9tETD_KGtmZQCyAHpP5IqlRK00yo2XMvZc-_mhjc-f1UWrY9OGwGh5vaBaP7xvsmZnU60Pp-A4eKoqjwucTx3mv9PzhkZR-ZzqeuEfSMBP082-Hxh7WuLOk_YuRMvHbKEzkzVU9-9h9kMeZWxfYFMVgAoyE1nd3o1gTjYupKsS1LOAdMJKe-6r75K4ceV_C4aUocXxLbPQ8j154lil5ujc0ejPvW709tWQINj7SdvnSb5zydKyGIsT-3eMGMthuWsNoCEKZB6JHnTYkeDsNzbvfwUqSexaH-eJM5MiFDPVeft-OG-OQrQxc8xrVXEOF3sGjLCgtbpmkvgcDDNYTKIByb9d0O0Mzmznz6HzWNPxhHRH8ZEvQEMPCRbxQn0UQoq-9UcHUVwoZUXl_w9kZ9zUgNVK8kNQKSikmKaWhAfI_NGz5zzqP_4-G8FfG1Vbdwt2l0g3okGtEUI7IU2hofXY4ypnIlmv_7dsjqkLsxhWChcO9BIdAyQ0svinQRW22b0ClgAQejDhpkJDwm8PJY03JKPgX9223GJ2zjXRO9Fx-OZV0TwWgt1xwhMbRenK6aJHDfnhsGZPQ5JjWAkMP_DQk8ZAGefCOHiYKdZZjpah0
    
    delta_iterate = previous_iterate - pre_previous_iterate
    delta_gradient = previous_gradient - pre_previous_gradient

    if (idx % 2 == 0):
        step_size = product.apply2(delta_iterate, delta_gradient) / product.apply2(delta_gradient, delta_gradient)
    else:
        step_size = product.apply2(delta_iterate, delta_iterate) / product.apply2(delta_iterate, delta_gradient)

    step_size = step_size[0,0]
    
    current_iterate = previous_iterate - step_size * search_direction

    if projector: 
        current_iterate = projector.project_domain(q, current_iterate) - q
        
    current_value = func(current_iterate)
    return (current_iterate, current_value)

def gradient_descent_linearized_problem(
    model : InstationaryModelIP,
    q : VectorArray,
    d_start : VectorArray,
    alpha : float,
    lin_solver_parms : Dict, 
    logger: logging.Logger = None,
    use_cached_operators: bool = False,
    projector: SimpleBoundDomainProjector = None) -> Tuple[VectorArray, int]:

    max_iter=lin_solver_parms['max_iter']
    lin_solver_tol=lin_solver_parms['lin_solver_tol']
    kappa_arm = lin_solver_parms['kappa_arm']
    armijo_inital_step_size = lin_solver_parms['armijo_inital_step_size']
    armijo_min_step_size = lin_solver_parms['armijo_min_step_size']

    assert alpha >= 0
    assert lin_solver_tol > 0
    assert kappa_arm > 0
    assert armijo_inital_step_size > 0
    assert armijo_min_step_size > 0

    if not logger:
        logger = logging.getLogger('gradient_descent')
        logger.setLevel(logging.DEBUG)

    previous_d = np.nan
    current_d = d_start

    previous_J = np.inf

    u = model.solve_state(q=q, use_cached_operators=use_cached_operators)
    lin_u = model.solve_linearized_state(q, current_d, u, use_cached_operators)
    current_J = model.linearized_objective(q, current_d, u, lin_u, alpha)
                                                   
    converged = False
    armijo_stagnation_flag = False
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

    def _compute_linearized_objective(d: VectorArray) -> float:
        lin_u = model.solve_linearized_state(q, d, u, use_cached_operators)    
        return model.linearized_objective(q, d, u, lin_u, alpha)
    

    for i in range(int(max_iter)):
        previous_d = current_d.copy()
        previous_J = current_J.copy()

        lin_u = model.solve_linearized_state(q, previous_d, u, use_cached_operators)    
        lin_p = model.solve_linearized_adjoint(q, u, lin_u, use_cached_operators)
        grad = model.linearized_gradient(q, previous_d, u, lin_p, alpha, use_cached_operators)

        buffer_nabla_J.pop(0)
        buffer_nabla_J.append(grad.copy())

        if projector:            
            terminaton_lhs = projector.project_domain(
                center = q,
                direction = -grad
            ) - q
        else:
            terminaton_lhs = -grad

        terminaton_lhs = model.compute_gradient_norm(terminaton_lhs)
        if (terminaton_lhs < lin_solver_tol) and i > 0:
            last_i = i + 1
            converged = True
            break

        if model.q_time_dep:
            product = model.products['bochner_prod_Q']
        else:
            product = model.products['prod_Q']

        # TODO Allow toggle between armijo and BB
        if i < 2:
            norm_grad = model.compute_gradient_norm(grad)            
            grad.scal(1.0 / norm_grad)
            current_d, current_J, armijo_stagnation_flag = armijo_line_serach(
                previous_iterate = previous_d,
                previous_value = previous_J,
                search_direction = -grad,
                func = _compute_linearized_objective,
                product=product,
                inital_step_size = armijo_inital_step_size,
                projector = projector,
                q=q,
                kappa_arm = kappa_arm,
                min_step_size = armijo_min_step_size)       
        else:
            current_d, current_J = barzilai_borwein_line_serach(
                previous_iterate =  buffer_d[-1],
                pre_previous_iterate = buffer_d[-2],
                previous_gradient = buffer_nabla_J[-1],
                pre_previous_gradient = buffer_nabla_J[-2],
                product=product,
                search_direction = grad,
                func = _compute_linearized_objective,
                projector = projector,
                q=q,
                idx=i)
            
        
        
        #if (i % 100 == 0):
        if (i % 1 == 0):
            logger.info(f"  Iteration {i+1} of {int(max_iter)} : objective = {current_J:3.4e}, norm gradient = {terminaton_lhs:3.4e}.")
            

        buffer_d.pop(0)
        buffer_d.append(current_d)

        # __d = current_d.to_numpy().reshape(9,9)
        # print(__d)
        # import matplotlib.pyplot as plt
        # plt.imshow(__d)
        # plt.savefig(f'./d_{i}.png')

        buffer_J.pop(0)
        buffer_J.append(current_J)    

        #stagnation check
        if armijo_stagnation_flag:
            logger.info(f"Stop at iteration {i+1} of {int(max_iter)}, due to stagnation (Armijo).")
            break

        if i > 5:
            if abs(buffer_J[0] - buffer_J[1]) < CONV_TOL and abs(buffer_J[1] - buffer_J[2]) < CONV_TOL:
                logger.info(f"Stop at iteration {i+1} of {int(max_iter)}, due to stagnation.")
                break
        
    if converged:
        logger.info(f"Gradient decent converged at iteration {last_i} of {int(max_iter)}.")
    else:
        logger.info(f"Gradient decent NOT converged after {int(max_iter)} iterations.")

    norm_grad = model.compute_gradient_norm(grad)
    logger.info(f"objective = {current_J:3.4e}, norm gradient = {norm_grad:3.4e}.")

    return current_d, last_i
