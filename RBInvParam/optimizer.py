import logging
import numpy as np
from abc import abstractmethod
from typing import Dict, Union, Tuple
from timeit import default_timer as timer

from pymor.vectorarrays.interface import VectorArray
from pymor.algorithms.hapod import inc_vectorarray_hapod
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.operators.interface import Operator

from RBInvParam.model import InstationaryModelIP
from RBInvParam.gradient_descent import gradient_descent_linearized_problem
from RBInvParam.reductor import InstationaryModelIPReductor
from RBInvParam.utils.logger import get_default_logger

MACHINE_EPS = 1e-16

# TODOs:
# - Fix logger for reductor
# - Use colors also in the log-files
# - Refactor AssembledB

class Optimizer:
    def __init__(self, 
                 optimizer_parameter: Dict, 
                 FOM : InstationaryModelIP,
                 logger: logging.Logger = None) -> None:
                 
        self.FOM = FOM
        self.optimizer_parameter = optimizer_parameter
        self._check_optimizer_parameter()    
        logging.basicConfig()

        # TODO Add class and function logger
        if logger:
            self.logger = logger
        else:
            self.logger = get_default_logger(self.__class__.__name__)
            self.logger.setLevel(logging.DEBUG)

        self.IRGNM_idx = 0


    
    def _check_optimizer_parameter(self) -> None:
        keys = self.optimizer_parameter.keys()

        assert self.optimizer_parameter['alpha_0'] >= 0
        assert self.optimizer_parameter['tol'] > 0
        assert self.optimizer_parameter['tau'] > 0
        assert self.optimizer_parameter['noise_level'] >= 0
        assert 0 < self.optimizer_parameter['theta'] \
                 < self.optimizer_parameter['Theta'] \
                 < 1
        if 'tau_tilde' in keys:
            assert self.optimizer_parameter['tau_tilde'] > 0

        assert self.optimizer_parameter['i_max'] >= 1
        if 'i_max_inner' in keys:
            assert self.optimizer_parameter['i_max_inner'] >= 1
        assert self.optimizer_parameter['reg_loop_max'] >= 1
        if 'armijo_max_iter' in keys:
            assert self.optimizer_parameter['armijo_max_iter'] >= 1

        if 'eta0' in keys:
            assert self.optimizer_parameter['eta0'] > 0
        if 'kappa_arm' in keys:
            assert self.optimizer_parameter['kappa_arm'] > 0
        if 'beta_1' in keys:
            assert 0 <= self.optimizer_parameter['beta_1'] < 1
        if 'beta_2' in keys:
            assert 3/4 <= self.optimizer_parameter['beta_2'] < 1
        if 'beta_3' in keys:
            assert 0 < self.optimizer_parameter['beta_3'] < 1    
    
    def IRGNM(self,
              model: InstationaryModelIP,
              q_0: VectorArray,
              alpha_0: float, 
              tol : float,
              tau : float,
              noise_level : float,
              theta: float, 
              Theta : float,
              i_max : int,
              reg_loop_max: int) -> Tuple[VectorArray, Dict]: 

        assert q_0 in model.Q
        assert tol > 0
        assert tau > 0
        assert noise_level >= 0
        assert 0 < theta < Theta < 1

        IRGNM_statistics = {
            'IRGNM_idx' : self.IRGNM_idx,
            'q' : [],
            'time_steps' : [],
            'alpha' : [],
            'J' : [],
            'norm_nabla_J' : [],
            'total_runtime' : np.nan,
            'stagnation_flag' : False,
        }
        start_time = timer()
        i = 0

        alpha = alpha_0
        q = q_0
        u = model.solve_state(q)
        p = model.solve_adjoint(q, u)
        J = model.objective(u)
        nabla_J = model.gradient(u, p)
        norm_nabla_J = model.compute_gradient_norm(nabla_J)

        IRGNM_statistics['q'].append(q)
        IRGNM_statistics['J'].append(J)
        IRGNM_statistics['norm_nabla_J'].append(norm_nabla_J)
        IRGNM_statistics['alpha'].append(alpha)

        self.logger.debug("Running IRGNM: ")
        self.logger.debug(f"  J : {J:3.4e}")
        self.logger.debug(f"  norm_nabla_J : {norm_nabla_J:3.4e}")
        self.logger.debug(f"                ")
        self.logger.debug(f"  alpha_0 : {alpha_0:3.4e}")
        self.logger.debug(f"  tol : {tol:3.4e}")
        self.logger.debug(f"  tau : {tau:3.4e}")
        self.logger.debug(f"  theta : {theta:3.4e}")
        self.logger.debug(f"  Theta : {Theta:3.4e}")
        self.logger.debug(f"                ")
        self.logger.debug(f"  i_max : {i_max:3.4e}")
        self.logger.debug(f"  reg_loop_max : {reg_loop_max:3.4e}")

        while J >= tol+tau*noise_level and i<i_max:
            self.logger.info(f"##############################################################################################################################")
            self.logger.warning(f"IRGNM: Iteration {i} | J = {J:3.4e} is not sufficent: {J:3.4e} > {(tol+tau*noise_level):3.4e}.")
            self.logger.info(f'Start IRGNM iteration {i}: J = {J:3.4e}, norm_nabla_J = {model.compute_gradient_norm(nabla_J):3.4e}, alpha = {alpha:1.4e}')
            self.logger.info(f"------------------------------------------------------------------------------------------------------------------------------")
            self.logger.info(f"Try 0: test alpha = {alpha:3.4e}.")

            regularization_qualification = False
            count = 1

            d_start = q.to_numpy().copy()
            d_start[:,:] = 0
            d_start = model.Q.make_array(d_start)

            max_iter = 1e4
            gc_tol = 1e-14
            inital_step_size = 1
            d = self.solve_linearized_problem(model=model,
                                              q=q,
                                              d_start=d_start,
                                              alpha=alpha,
                                              method='gd',
                                              max_iter=max_iter,
                                              tol=gc_tol,
                                              inital_step_size=inital_step_size, 
                                              logger = self.logger)
            
            lin_u = model.solve_linearized_state(q, d, u)
            lin_J = model.linearized_objective(q, d, u, lin_u, alpha=0)

            condition_low = theta*J< 2*lin_J
            condition_up = 2* lin_J < Theta*J
            regularization_qualification = condition_low and condition_up

            if (not regularization_qualification) and (count < reg_loop_max):
                self.logger.warning(f"Used alpha = {alpha:3.4e} does NOT satisfy selection criteria: {theta*J:3.4e} < {2* lin_J:3.4e} < {Theta*J:3.4e}")
                self.logger.info(f"Searching for alpha:") 

            loop_terminated = False
            while (not regularization_qualification) and (count < reg_loop_max):

                if alpha <= 1e-14:
                    loop_terminated = True
                    break
                
                if not condition_low:
                    alpha *= 1.5  
                elif not condition_up:
                    alpha = max(alpha/2,1e-14)
                else:
                    raise ValueError
                
                self.logger.info(f"------------------------------------------------------------------------------------------------------------------------------")
                self.logger.info(f"Try {count}: test alpha = {alpha:3.4e}.")
                d = self.solve_linearized_problem(model=model,
                                                  q=q,
                                                  d_start=d_start,
                                                  alpha=alpha,
                                                  method='gd',
                                                  max_iter=max_iter,
                                                  tol=gc_tol,
                                                  inital_step_size=inital_step_size,
                                                  logger = self.logger)

                lin_u = model.solve_linearized_state(q, d, u)
                lin_J = model.linearized_objective(q, d, u, lin_u, alpha=0)

                condition_low = theta*J< 2 * lin_J
                condition_up = 2* lin_J < Theta*J
                regularization_qualification = condition_low and condition_up
                            
                if (not regularization_qualification) and (count < reg_loop_max):
                    self.logger.warning(f"Used alpha = {alpha:3.4e} does NOT satisfy selection criteria: {theta*J:3.4e} < {2* lin_J:3.4e} < {Theta*J:3.4e}")
                else:
                    self.logger.info(f"------------------------------------------------------------------------------------------------------------------------------")

                count += 1

            loop_terminated = loop_terminated or (count >= reg_loop_max)

            if not loop_terminated:
                self.logger.warning(f"Used alpha = {alpha:3.4e} does satisfy selection criteria: {theta*J:3.4e} < {2* lin_J:3.4e} < {Theta*J:3.4e}")
            else:
                self.logger.error(f"Not found valid alpha before reaching maximum number of tries : {reg_loop_max}.\n\
                                   Using the last alpha tested = {alpha:3.4e}.")
                
            ########################################### Armijo ###########################################
            # TODO 
            ########################################### Final ###########################################


            q += d
            u = model.solve_state(q)
            p = model.solve_adjoint(q, u)
            J = model.objective(u)
            nabla_J = model.gradient(u, p)
            norm_nabla_J = model.compute_gradient_norm(nabla_J)

            IRGNM_statistics['q'].append(q)
            IRGNM_statistics['J'].append(J)
            IRGNM_statistics['norm_nabla_J'].append(norm_nabla_J)
            IRGNM_statistics['alpha'].append(alpha)
        
            #stagnation check
            if i > 3:
                buffer = IRGNM_statistics['J'][-3:]
                if abs(buffer[0] - buffer[1]) < MACHINE_EPS and abs(buffer[1] -buffer[2]) < MACHINE_EPS:
                    IRGNM_statistics['stagnation_flag'] = True
                    self.logger.info(f"Stop at iteration {i+1} of {int(i_max)}, due to stagnation.")
                    break

            IRGNM_statistics['time_steps'].append((timer()- start_time))
            self.logger.info(f'Statistics IRGNM iteration {i}: J = {J:3.4e}, norm_nabla_J = {norm_nabla_J:3.4e}, alpha = {alpha:1.4e}')
            i += 1
            if not(J >= tol+tau*noise_level and i<i_max):
                self.logger.info(f"##############################################################################################################################")

        IRGNM_statistics['total_runtime'] = (timer() - start_time)        
        return (q, IRGNM_statistics)        

    def solve_linearized_problem(self,
                                model : InstationaryModelIP, 
                                q : np.array,
                                d_start : np.array,
                                alpha : float,
                                method : str,
                                **kwargs : Dict) -> np.array:
    
        if method == 'gd':
            return gradient_descent_linearized_problem(model, q, d_start, alpha, **kwargs)
        elif method == 'cg':
            raise NotImplementedError
        else:
            raise ValueError
    
    def _HaPOD(self, 
               shapshots: VectorArray, 
               basis: str,
               product: Operator) -> Tuple[VectorArray, np.array]:
            
        if len(self.reductor.bases[basis]) != 0:
            projected_shapshots = self.reductor.bases[basis].lincomb(
                self.reductor.project_vectorarray(shapshots, basis=basis)
            )
            shapshots.axpy(-1,projected_shapshots)
                
        shapshots, svals, _ = \
        inc_vectorarray_hapod(steps=len(shapshots)/5, 
                              U=shapshots, 
                              eps=1e-16,
                              omega=0.1,                
                              product=product)

        return shapshots, svals

class FOMOptimizer(Optimizer):
    def __init__(self, 
                 optimizer_parameter: Dict, 
                 FOM : InstationaryModelIP,
                 logger: logging.Logger = None) -> None:

        super().__init__(optimizer_parameter, FOM, logger)
        self.statistics = {
            'q' : [],
            'time_steps' : [],
            'alpha' : [],
            'J' : [],
            'norm_nabla_J' : [],
            'total_runtime' : np.nan,
            'stagnation_flag' : False,
            'optimizer_parameter' : self.optimizer_parameter.copy()
        }

    def solve(self) -> VectorArray:
        self.IRGNM_idx += 1
        q, IRGNM_statistic = self.IRGNM(model = self.FOM,
                                        q_0 = self.FOM.Q.make_array(self.optimizer_parameter['q_0'].copy()),
                                        alpha_0 = self.optimizer_parameter['alpha_0'],
                                        tol = self.optimizer_parameter['tol'],
                                        tau = self.optimizer_parameter['tau'],
                                        noise_level = self.optimizer_parameter['noise_level'],
                                        theta = self.optimizer_parameter['theta'],
                                        Theta = self.optimizer_parameter['Theta'],
                                        i_max = self.optimizer_parameter['i_max'],
                                        reg_loop_max = self.optimizer_parameter['reg_loop_max'])

        self.statistics['q'] = IRGNM_statistic['q']
        self.statistics['time_steps'] = IRGNM_statistic['time_steps']
        self.statistics['alpha'] = IRGNM_statistic['alpha']
        self.statistics['J'] = IRGNM_statistic['J']
        self.statistics['norm_nabla_J'] = IRGNM_statistic['norm_nabla_J']
        self.statistics['total_runtime'] = IRGNM_statistic['total_runtime']
        self.statistics['stagnation_flag'] = IRGNM_statistic['stagnation_flag']

        return q
        
class QrFOMOptimizer(Optimizer):
    def __init__(self, 
                 optimizer_parameter: Dict, 
                 FOM : InstationaryModelIP,
                 logger: logging.Logger = None) -> None:

        super().__init__(optimizer_parameter, FOM, logger)
        self.reductor = InstationaryModelIPReductor(
            FOM
        )
        self.QrFOM = None

        self.statistics = {
            'q' : [],
            'inner_loop_time_steps' : [],
            'alpha' : [],
            'J' : [],
            'norm_nabla_J' : [],
            'total_runtime' : np.nan,
            'stagnation_flag' : False,
            'optimizer_parameter' : self.optimizer_parameter.copy()
        }
    

    def solve(self) -> VectorArray:
        q_0 = self.optimizer_parameter['q_0'].copy()
        alpha_0 = self.optimizer_parameter['alpha_0']
        tol = self.optimizer_parameter['tol']
        tau = self.optimizer_parameter['tau']
        noise_level = self.optimizer_parameter['noise_level']
        theta = self.optimizer_parameter['theta']
        Theta = self.optimizer_parameter['Theta']

        i_max = self.optimizer_parameter['i_max']
        reg_loop_max = self.optimizer_parameter['reg_loop_max']
        i_max_inner = self.optimizer_parameter['i_max_inner']

        
        start_time = timer()
        i = 0
        alpha = alpha_0
        delta = noise_level

        q = self.FOM.Q.make_array(q_0)
        u = self.FOM.solve_state(q)
        p = self.FOM.solve_adjoint(q, u)
        J = self.FOM.objective(u)
        nabla_J = self.FOM.gradient(u, p)
        norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)
        
        self.statistics['q'].append(q)
        self.statistics['alpha'].append(alpha)
        self.statistics['J'].append(J)
        self.statistics['norm_nabla_J'].append(norm_nabla_J)


        self.logger.debug("Running Qr-IRGNM:")
        self.logger.debug(f"  J : {J:3.4e}")
        self.logger.debug(f"  norm_nabla_J : {norm_nabla_J:3.4e}")
        self.logger.debug(f"                ")
        self.logger.debug(f"  alpha_0 : {alpha_0:3.4e}")
        self.logger.debug(f"  tol : {tol:3.4e}")
        self.logger.debug(f"  tau : {tau:3.4e}")
        self.logger.debug(f"  noise_level : {noise_level:3.4e}")
        self.logger.debug(f"  theta : {theta:3.4e}")
        self.logger.debug(f"  Theta : {Theta:3.4e}")
        self.logger.debug(f"                ")
        self.logger.debug(f"  i_max : {i_max:3.4e}")
        self.logger.debug(f"  i_max_inner : {i_max_inner:3.4e}")
        self.logger.debug(f"  reg_loop_max : {reg_loop_max:3.4e}")
        
        self.logger.debug(f"Extending Qr-space")
        parameter_shapshots = self.FOM.Q.empty()
        parameter_shapshots.append(nabla_J)
        parameter_shapshots.append(q)
        parameter_shapshots.append(self.FOM.Q.make_array(self.FOM.setup['model_parameter']['q_circ']))

        if self.FOM.setup['model_parameter']['q_time_dep']:
            self.logger.debug(f"Performing HaPOD on parameter snapshots.")
            parameter_shapshots, _ = self._HaPOD(shapshots=parameter_shapshots, 
                                                 basis='parameter_basis',
                                                 product=self.FOM.products['prod_Q'])

        self.reductor.extend_basis(
             U = parameter_shapshots,
             basis = 'parameter_basis'
        )
        self.QrFOM = self.reductor.reduce()

        self.logger.debug(f"Dim Qr-space = {self.reductor.get_bases_dim('parameter_basis')}")
        self.logger.debug(f"Dim Vr-space = {self.reductor.get_bases_dim('state_basis')}")

        while J >= tol+tau*noise_level and i<i_max:
            self.logger.info(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            self.logger.warning(f"Qr-IRGNM iteration {i}: J = {J:3.4e} is not sufficent: {J:3.4e} > {(tol+tau*noise_level):3.4e}.")
            self.logger.info(f'Start Qr-IRGNM iteration {i}: J = {J:3.4e}, norm_nabla_J = {self.FOM.compute_gradient_norm(nabla_J):3.4e}, alpha = {alpha:1.4e}')
            self.logger.info(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            q_r = self.reductor.project_vectorarray(q, 'parameter_basis')
            q_r = self.QrFOM.Q.make_array(q_r)

            q_r, IRGNM_statistic = self.IRGNM(model = self.QrFOM,
                                              q_0 = q_r,
                                              alpha_0 = alpha,
                                              tol = tol,
                                              tau = tau,
                                              noise_level = delta,
                                              i_max = i_max_inner,
                                              theta = theta,
                                              Theta = Theta,
                                              reg_loop_max = reg_loop_max)
            

            q = self.reductor.reconstruct(q_r, basis='parameter_basis')
            u = self.FOM.solve_state(q)
            p = self.FOM.solve_adjoint(q, u)
            J = self.FOM.objective(u)
            nabla_J = self.FOM.gradient(u, p)
            alpha = IRGNM_statistic['alpha'][1]

            self.statistics['q'].append(q)
            self.statistics['alpha'].append(alpha)
            self.statistics['J'].append(J)
            self.statistics['norm_nabla_J'].append(self.FOM.compute_gradient_norm(nabla_J))

            if i > 3:
                buffer = self.statistics['J'][-3:]
                if abs(buffer[0] - buffer[1]) < MACHINE_EPS and abs(buffer[1] - buffer[2]) < MACHINE_EPS:
                    self.statistics['stagnation_flag'] = True
                    self.logger.info(f"Stop at iteration {i+1} of {int(i_max)}, due to stagnation.")
                    break
            
            self.logger.debug(f"Extending Qr-space")
            parameter_shapshots = self.FOM.Q.empty()
            parameter_shapshots.append(nabla_J)
            
            if self.FOM.setup['model_parameter']['q_time_dep']:
                self.logger.debug(f"Performing HaPOD on parameter snapshots.")
                parameter_shapshots, _ = self._HaPOD(shapshots=parameter_shapshots, 
                                                     basis='parameter_basis',
                                                     product=self.FOM.products['prod_Q'])

            self.reductor.extend_basis(
                U = parameter_shapshots,
                basis = 'parameter_basis'
            )
            self.QrFOM = self.reductor.reduce()
            self.logger.debug(f"Dim Qr-space = {self.reductor.get_bases_dim('parameter_basis')}")
            self.logger.debug(f"Dim Vr-space = {self.reductor.get_bases_dim('state_basis')}")

        self.statistics['total_runtime'] = (timer() - start_time)        
        return q
        
class QrVrROMOptimizer(Optimizer):
    def __init__(self, 
                 optimizer_parameter: Dict, 
                 FOM : InstationaryModelIP,
                 logger: logging.Logger = None) -> None:

        super().__init__(optimizer_parameter, FOM, logger)
        self.reductor = InstationaryModelIPReductor(
            FOM
        )
        self.QrVrROM = None

        self.statistics = {
            'q' : [],
            'inner_loop_time_steps' : [],
            'alpha' : [],
            'J' : [],
            'norm_nabla_J' : [],
            'total_runtime' : np.nan,
            'stagnation_flag' : False,
            'optimizer_parameter' : self.optimizer_parameter.copy()
        }

    def armijo_TR_line_serach(self,
                              model: InstationaryModelIP, 
                              previous_q: NumpyVectorArray,
                              previous_J: float,
                              search_direction : NumpyVectorArray,
                              max_iter: int,
                              inital_step_size: float,
                              eta: float,
                              beta: float,
                              kappa_arm: float) -> Tuple[NumpyVectorArray, float, bool]:
        
        assert 0 <= beta < 1
        assert 0 < eta
        i = 0
        model_unsufficent = False

        self.logger.info(f"Start Armijo backtracking, with J = {previous_J:3.4e}.")

        step_size = inital_step_size
        search_direction.scal(1.0 / model.compute_gradient_norm(search_direction))
        current_q = previous_q + step_size * search_direction
        u = model.solve_state(q=current_q)
        p = model.solve_adjoint(q=current_q, u=u)
        current_J = model.objective(u=u,
                                    q=current_q)
        
        norm_d = model.compute_gradient_norm(previous_q - current_q)
        lhs =  previous_J - current_J
        rhs = kappa_arm / step_size * norm_d**2
        
        if abs(lhs) <= MACHINE_EPS:
            lhs = 0

        if abs(rhs) <= MACHINE_EPS:
            rhs = 0

        armijo_condition = lhs >= rhs
        #armijo_condition = lhs <= rhs
        if current_J > 0:
            J_rel_error = model.estimate_objective_error(
                q=current_q,
                u = u,
                p = p) / current_J
        else:
            J_rel_error = np.inf
        
        TR_condition = J_rel_error <= eta
        condition = armijo_condition & TR_condition

        while (not condition) and (i <= max_iter) :
            step_size = 0.5 * step_size
            current_q = previous_q + step_size * search_direction
            u = model.solve_state(q=current_q)
            p = model.solve_adjoint(q=current_q, u=u)
            current_J = model.objective(u=u,
                                        q=current_q)
            
            norm_d = model.compute_gradient_norm(previous_q - current_q)
            lhs = previous_J - current_J
            rhs = kappa_arm / step_size * norm_d**2
            
            if abs(lhs) <= MACHINE_EPS:
                lhs = 0

            if abs(rhs) <= MACHINE_EPS:
                rhs = 0

            armijo_condition = lhs >= rhs
            #armijo_condition = lhs <= rhs

            if current_J > 0:
                J_rel_error = model.estimate_objective_error(
                    q=current_q,
                    u = u,
                    p = p) / current_J
            else:
                J_rel_error = np.inf

            TR_condition = J_rel_error <= eta
            condition = armijo_condition & TR_condition
            
            i += 1

        if (J_rel_error >= beta * eta) or (i == max_iter):
            model_unsufficent = True


        if not condition:
            self.logger.error(f"Armijo backtracking does NOT terminate normally. step_size = {step_size:3.4e}; Stopping at J = {current_J:3.4e}")
            self.logger.debug(f"armijo_condition = {armijo_condition}, TR_condition = {TR_condition}")

        else:
            self.logger.debug(f"Armijo backtracking does terminate normally with step_size = {step_size:3.4e}; Stopping at J = {current_J:3.4e}")
            

        return (current_q, current_J, model_unsufficent)


    def solve(self) -> VectorArray :
        q_0 = self.optimizer_parameter['q_0'].copy()
        alpha_0 = self.optimizer_parameter['alpha_0']
        tol = self.optimizer_parameter['tol']
        tau = self.optimizer_parameter['tau']
        noise_level = self.optimizer_parameter['noise_level']
        theta = self.optimizer_parameter['theta']
        Theta = self.optimizer_parameter['Theta']
        tau_tilde = self.optimizer_parameter['tau_tilde']

        i_max = self.optimizer_parameter['i_max']
        reg_loop_max = self.optimizer_parameter['reg_loop_max']
        i_max_inner = self.optimizer_parameter['i_max_inner']
        armijo_max_iter = self.optimizer_parameter['armijo_max_iter']

        eta0 = self.optimizer_parameter['eta0']
        kappa_arm = self.optimizer_parameter['kappa_arm']
        beta_1 = self.optimizer_parameter['beta_1']
        beta_2 = self.optimizer_parameter['beta_2']
        beta_3 = self.optimizer_parameter['beta_3']

        start_time = timer()
        i = 0
        alpha = alpha_0
        delta = noise_level

        q = self.FOM.Q.make_array(q_0)
        u = self.FOM.solve_state(q)
        p = self.FOM.solve_adjoint(q, u)
        J = self.FOM.objective(u)
        nabla_J = self.FOM.gradient(u, p)
        norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)
        assert norm_nabla_J > 0

        inital_armijo_step_size = 0.5 / norm_nabla_J
        eta = eta0
        
        self.statistics['q'].append(q)
        self.statistics['alpha'].append(alpha)
        self.statistics['J'].append(J)
        self.statistics['norm_nabla_J'].append(norm_nabla_J)

        self.logger.debug("Running Qr-Vr-IRGNM:")
        self.logger.debug(f"  J : {J:3.4e}")
        self.logger.debug(f"  norm_nabla_J : {norm_nabla_J:3.4e}")
        self.logger.debug(f"                ")
        self.logger.debug(f"  alpha_0 : {alpha_0:3.4e}")
        self.logger.debug(f"  tol : {tol:3.4e}")
        self.logger.debug(f"  tau : {tau:3.4e}")
        self.logger.debug(f"  noise_level : {noise_level:3.4e}")
        self.logger.debug(f"  theta : {theta:3.4e}")
        self.logger.debug(f"  Theta : {Theta:3.4e}")
        self.logger.debug(f"  tau_tilde : {tau_tilde:3.4e}")
        self.logger.debug(f"                ")
        self.logger.debug(f"  i_max : {i_max:3.4e}")
        self.logger.debug(f"  i_max_inner : {i_max_inner:3.4e}")
        self.logger.debug(f"  reg_loop_max : {reg_loop_max:3.4e}")
        self.logger.debug(f"  armijo_max_iter : {armijo_max_iter:3.4e}")
        self.logger.debug(f"                ")
        self.logger.debug(f"  eta0 : {eta0:3.4e}")
        self.logger.debug(f"  kappa_arm : {kappa_arm:3.4e}")
        self.logger.debug(f"  beta_1 : {beta_1:3.4e}")
        self.logger.debug(f"  beta_2 : {beta_2:3.4e}")
        self.logger.debug(f"  beta_3 : {beta_3:3.4e}")


        self.logger.debug(f"Extending Qr-space")
        parameter_shapshots = self.FOM.Q.empty()
        parameter_shapshots.append(nabla_J)
        parameter_shapshots.append(q)
        parameter_shapshots.append(self.FOM.Q.make_array(self.FOM.setup['model_parameter']['q_circ']))
        
        if self.FOM.setup['model_parameter']['q_time_dep']:
            self.logger.debug(f"Performing HaPOD on parameter snapshots.")
            parameter_shapshots, _ = self._HaPOD(shapshots=parameter_shapshots, 
                                                 basis='parameter_basis',
                                                 product=self.FOM.products['prod_Q'])

        self.reductor.extend_basis(
             U = parameter_shapshots,
             basis = 'parameter_basis'
        )

        self.logger.debug(f"Extending Vr-space")
        state_shapshots = self.FOM.V.empty()
        state_shapshots.append(u)
        state_shapshots.append(p)

        self.logger.debug(f"Performing HaPOD on state snapshots.")
        parameter_shapshots, _ = self._HaPOD(shapshots=state_shapshots, 
                                             basis='state_basis',
                                             product=self.FOM.products['prod_V'])
        
        self.reductor.extend_basis(
             U = state_shapshots,
             basis = 'state_basis'
        )

        self.QrVrROM = self.reductor.reduce()
        
        self.logger.debug(f"Dim Qr-space = {self.reductor.get_bases_dim('parameter_basis')}")
        self.logger.debug(f"Dim Vr-space = {self.reductor.get_bases_dim('state_basis')}")

        q_r = self.reductor.project_vectorarray(q, 'parameter_basis')
        q_r = self.QrVrROM.Q.make_array(q_r)

        while J >= tol+tau*noise_level and i<i_max:
            self.logger.info(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            self.logger.warning(f"Qr-Vr-IRGNM iteration {i}: J = {J:3.4e} is not sufficent: {J:3.4e} > {(tol+tau*noise_level):3.4e}.")
            self.logger.info(f'Start Qr-Vr-IRGNM iteration {i}: J = {J:3.4e}, norm_nabla_J = {self.FOM.compute_gradient_norm(nabla_J):3.4e}, alpha = {alpha:1.4e}')
            self.logger.info(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            q_r = self.reductor.project_vectorarray(q, 'parameter_basis')
            q_r = self.QrVrROM.Q.make_array(q_r)

            u_r = self.QrVrROM.solve_state(q_r)
            p_r = self.QrVrROM.solve_adjoint(q_r, u_r)
            J_r = self.QrVrROM.objective(u_r)
            nabla_J_r = self.QrVrROM.gradient(u_r, p_r)
            

            ########################################### AGC ###########################################

            self.logger.warning("Calculate AGC with Armijo backtracking.")
            q_r, _, model_unsufficent = self.armijo_TR_line_serach(
                model = self.QrVrROM,
                previous_q = q_r,
                previous_J = J_r,
                search_direction = -nabla_J_r,
                max_iter = armijo_max_iter,
                inital_step_size = inital_armijo_step_size,
                eta = eta,
                beta = beta_1,
                kappa_arm = kappa_arm,
            )

            ########################################### IRGNM ###########################################
            q_r, IRGNM_statistic = self.IRGNM(model = self.QrVrROM,
                                              q_0 = q_r,
                                              alpha_0 = alpha,
                                              tol = tol,
                                              tau = tau,
                                              noise_level = delta,
                                              i_max = i_max_inner,
                                              theta = theta,
                                              Theta = Theta,
                                              reg_loop_max = reg_loop_max)            
            ########################################### Accept / Reject ###########################################
            if False: 
                pass
            elif False:
                pass
                # TODO eta
            else:
                if False:
                    pass
                else:
                    pass

            ########################################### Final ###########################################
            q = self.reductor.reconstruct(q_r, basis='parameter_basis')
            u = self.FOM.solve_state(q)
            p = self.FOM.solve_adjoint(q, u)
            J = self.FOM.objective(u)
            nabla_J = self.FOM.gradient(u, p)
            alpha = IRGNM_statistic['alpha'][1]

            self.statistics['q'].append(q)
            self.statistics['alpha'].append(alpha)
            self.statistics['J'].append(J)
            self.statistics['norm_nabla_J'].append(self.FOM.compute_gradient_norm(nabla_J))

            if i > 3:
                buffer = self.statistics['J'][-3:]
                if abs(buffer[0] - buffer[1]) < MACHINE_EPS and abs(buffer[1] - buffer[2]) < MACHINE_EPS:
                    self.statistics['stagnation_flag'] = True
                    self.logger.info(f"Stop at iteration {i+1} of {int(i_max)}, due to stagnation.")
                    break

            self.logger.debug(f"Extending Qr-space")
            parameter_shapshots = self.FOM.Q.empty()
            parameter_shapshots.append(nabla_J)

            if self.FOM.setup['model_parameter']['q_time_dep']:
                self.logger.debug(f"Performing HaPOD on parameter snapshots.")
                parameter_shapshots, _ = self._HaPOD(shapshots=parameter_shapshots, 
                                                     basis='parameter_basis',
                                                     product=self.FOM.products['prod_Q'])
            self.reductor.extend_basis(
                U = parameter_shapshots,
                basis = 'parameter_basis'
            )
            self.logger.debug(f"Extending Vr-space")
            state_shapshots = self.FOM.V.empty()
            state_shapshots.append(u)
            state_shapshots.append(p)

            self.logger.debug(f"Performing HaPOD on state snapshots.")
            parameter_shapshots, _ = self._HaPOD(shapshots=state_shapshots, 
                                                 basis='state_basis',
                                                 product=self.FOM.products['prod_V'])
            
            self.reductor.extend_basis(
                U = state_shapshots,
                basis = 'state_basis'
            )

            self.QrVrROM = self.reductor.reduce()
            self.logger.debug(f"Dim Qr-space = {self.reductor.get_bases_dim('parameter_basis')}")
            self.logger.debug(f"Dim Vr-space = {self.reductor.get_bases_dim('state_basis')}")

            i += 1

        self.statistics['total_runtime'] = (timer() - start_time)
        return q
