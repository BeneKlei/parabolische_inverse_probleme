import numpy as np
import logging
from typing import Callable, Tuple, Dict
from functools import partial

from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.vectorarrays.interface import VectorArray
from pymor.operators.numpy import NumpyMatrixOperator

from RBInvParam.model import InstationaryModelIP
from RBInvParam.reductor import InstationaryModelIPReductor


MACHINE_EPS = 1e-16

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
    rhs = kappa_arm / step_size * norm_d**2
    

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
                       projector: Callable = None,
                       q: NumpyVectorArray = None) -> Tuple[NumpyVectorArray, float]:

        step_size = inital_step_size
        current_iterate = previous_iterate + step_size * search_direction

        if projector: 
            current_iterate = projector(q + current_iterate) - q

        current_value = func(current_iterate)

        condition = armijo_condition(previous_value, 
                                     current_value, 
                                     step_size, 
                                     previous_iterate, 
                                     current_iterate,
                                     product=product,
                                     kappa_arm=1e-12)

        while not condition:
            step_size = 0.5 * step_size
            current_iterate = previous_iterate + step_size * search_direction

            if projector: 
                current_iterate = projector(q + current_iterate) - q


            current_value = func(current_iterate)

            condition = armijo_condition(previous_value, 
                                         current_value, 
                                         step_size, 
                                         previous_iterate, 
                                         current_iterate,
                                         product=product,
                                         kappa_arm=1e-12)

        return (current_iterate, current_value)

def barzilai_borwein_line_serach(previous_iterate: NumpyVectorArray,
                                 pre_previous_iterate: NumpyVectorArray,
                                 previous_gradient: NumpyVectorArray,
                                 pre_previous_gradient: NumpyVectorArray,
                                 product : NumpyMatrixOperator,
                                 search_direction : NumpyVectorArray,
                                 func: Callable, 
                                 projector: Callable = None,
                                 q: NumpyVectorArray = None) -> Tuple[NumpyVectorArray, float]:
    
    # Using algorithm from
    # https://watermark.silverchair.com/8-1-141.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA2kwggNlBgkqhkiG9w0BBwagggNWMIIDUgIBADCCA0sGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM4WESSjzDIsa2gVMsAgEQgIIDHHeiS0AQ9L_1TmZFA7rg1QFF7cezo4BC_1OnBjkdHDIidNO_gGqaOBss8MNgz6cK5xd1mDqhTB0w0Jx202D40CtChSI6QCQUfSpoFcR3D28U6jRYZnaH9NjLIh0rE59cktmXZbX0aCov-NgLpmyfyrWVhK0hdkl5aXU_2hrh3b83hg2wjA_k9JVXwxHDhaS58iAtIv8Ulw4jBc8E6iV447KcH4RKuUT8PISqwQoTWF5-5564fSGEYVWrV2SFDbiHQpqnJBSLTJdXMK5EwqxXN3Z7b7byHqUe76bZdk5f2RROTuMX2TRITeGRZdnyqQ2qL2O2lmqqOrjomiKg3qUYcX_2wqBOyD2WC3cIjHalwNEgPZfRVAqJ-UCrBsnBdcwIPDlATYhN3XG-zMBUKURfQt8ypcoPlYQoZD0NI-d2Hsr7-Bx7ishcO8tJ07hY9tETD_KGtmZQCyAHpP5IqlRK00yo2XMvZc-_mhjc-f1UWrY9OGwGh5vaBaP7xvsmZnU60Pp-A4eKoqjwucTx3mv9PzhkZR-ZzqeuEfSMBP082-Hxh7WuLOk_YuRMvHbKEzkzVU9-9h9kMeZWxfYFMVgAoyE1nd3o1gTjYupKsS1LOAdMJKe-6r75K4ceV_C4aUocXxLbPQ8j154lil5ujc0ejPvW709tWQINj7SdvnSb5zydKyGIsT-3eMGMthuWsNoCEKZB6JHnTYkeDsNzbvfwUqSexaH-eJM5MiFDPVeft-OG-OQrQxc8xrVXEOF3sGjLCgtbpmkvgcDDNYTKIByb9d0O0Mzmznz6HzWNPxhHRH8ZEvQEMPCRbxQn0UQoq-9UcHUVwoZUXl_w9kZ9zUgNVK8kNQKSikmKaWhAfI_NGz5zzqP_4-G8FfG1Vbdwt2l0g3okGtEUI7IU2hofXY4ypnIlmv_7dsjqkLsxhWChcO9BIdAyQ0svinQRW22b0ClgAQejDhpkJDwm8PJY03JKPgX9223GJ2zjXRO9Fx-OZV0TwWgt1xwhMbRenK6aJHDfnhsGZPQ5JjWAkMP_DQk8ZAGefCOHiYKdZZjpah0

    # print(previous_iterate)
    # print(pre_previous_iterate)
    
    delta_iterate = previous_iterate - pre_previous_iterate
    delta_gradient = previous_gradient - pre_previous_gradient

    #step_size = product.apply2(delta_iterate, delta_iterate) / product.apply2(delta_iterate, delta_gradient)
    step_size = product.apply2(delta_iterate, delta_gradient) / product.apply2(delta_gradient, delta_gradient)
    
    current_iterate = previous_iterate - step_size[0,0] * search_direction

    if projector: 
        current_iterate = projector(q + current_iterate) - q

    current_value = func(current_iterate)
    return (current_iterate, current_value)

def project_to_simple_domain(q: NumpyVectorArray,
                             model: InstationaryModelIP,
                             reductor: InstationaryModelIPReductor, 
                             bounds: np.ndarray) -> NumpyVectorArray:
    
    q_recon = reductor.reconstruct(q, basis='parameter_basis')
    q_recon = q_recon.to_numpy().flatten()
    
    mask_lb = q_recon < bounds[:,0]
    mask_ub = q_recon > bounds[:,1]

    q_recon[mask_lb] = bounds[mask_lb,0]
    q_recon[mask_ub] = bounds[mask_ub,1]

    q_recon = q_recon.reshape((reductor.FOM.nt, reductor.FOM.Q.dim))
    q_recon = reductor.FOM.Q.make_array(q_recon)  
    q_recon = reductor.project_vectorarray(q_recon, basis='parameter_basis')

    return model.Q.make_array(q_recon)

def gradient_descent_linearized_problem(
    model : InstationaryModelIP,
    q : VectorArray,
    d_start : VectorArray,
    alpha : float,
    lin_solver_parms : Dict, 
    logger: logging.Logger = None,
    use_cached_operators: bool = False,
    bounds : np.ndarray = None, 
    reductor : InstationaryModelIPReductor = None,) -> Tuple[VectorArray, int]:

    max_iter=lin_solver_parms['max_iter']
    lin_solver_tol=lin_solver_parms['lin_solver_tol']
    inital_step_size =lin_solver_parms['inital_step_size']

    assert alpha >= 0
    assert lin_solver_tol > 0
    assert inital_step_size > 0

    if not logger:
        logger = logging.getLogger('gradient_descent')
        logger.setLevel(logging.DEBUG)

    if bounds is not None:
        assert isinstance(bounds, np.ndarray)
        assert reductor

        if model.q_time_dep:
            assert bounds.shape == (model.nt * reductor.FOM.Q.dim , 2)
        else:
            assert bounds.shape == (reductor.FOM.Q.dim , 2)
        assert np.all(bounds[:,0] < bounds[:,1])

        projector = partial(project_to_simple_domain,
            model=model,
            reductor = reductor,
            bounds = bounds
        )
    else:
        projector = None

    previous_d = np.nan
    current_d = d_start

    previous_J = np.inf
    current_J = model.compute_linearized_objective(q, 
                                                   current_d, 
                                                   alpha, 
                                                   use_cached_operators=use_cached_operators)
                                                   
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

        grad = model.compute_linearized_gradient(q, 
                                                 previous_d, 
                                                 alpha, 
                                                 use_cached_operators=use_cached_operators)


        buffer_nabla_J.pop(0)
        buffer_nabla_J.append(grad.copy())
        norm_grad = model.compute_gradient_norm(grad)

        if norm_grad < lin_solver_tol:
            last_i = i + 1
            converged = True
            break

        if model.q_time_dep:
            product = model.products['bochner_prod_Q']
        else:
            product = model.products['prod_Q']

        # TODO Allow toggle between armijo and BB
        if i < 2:
            grad.scal(1.0 / norm_grad)
            current_d, current_J = armijo_line_serach(
                previous_iterate = previous_d,
                previous_value = previous_J,
                search_direction = -grad,
                func = lambda d: model.compute_linearized_objective(q, 
                                                                    d, 
                                                                    alpha, 
                                                                    use_cached_operators=use_cached_operators),
                product=product,
                inital_step_size = inital_step_size,
                projector = projector,
                q=q)       

        else:
            current_d, current_J = barzilai_borwein_line_serach(
                previous_iterate =  buffer_d[-1],
                pre_previous_iterate = buffer_d[-2],
                previous_gradient = buffer_nabla_J[-1],
                pre_previous_gradient = buffer_nabla_J[-2],
                product=product,
                search_direction = grad,
                func = lambda d: model.compute_linearized_objective(q, 
                                                                    d, 
                                                                    alpha, 
                                                                    use_cached_operators=use_cached_operators),
                projector = projector,
                q=q)
            
        
        
        if (i % 250 == 0):
            logger.info(f"  Iteration {i+1} of {int(max_iter)} : objective = {current_J:3.4e}, norm gradient = {norm_grad:3.4e}.")

        buffer_d.pop(0)
        buffer_d.append(current_d)

        buffer_J.pop(0)
        buffer_J.append(current_J)    

        #stagnation check
        if i > 5:
            if abs(buffer_J[0] - buffer_J[1]) < MACHINE_EPS and abs(buffer_J[1] - buffer_J[2]) < MACHINE_EPS:
                logger.info(f"Stop at iteration {i+1} of {int(max_iter)}, due to stagnation.")
                break

    if converged:
        logger.info(f"Gradient decent converged at iteration {last_i} of {int(max_iter)}.")
    else:
        logger.info(f"Gradient decent NOT converged after {int(max_iter)} iterations.")

    logger.info(f"objective = {current_J:3.4e}, norm gradient = {norm_grad:3.4e}.")

    return current_d, last_i
