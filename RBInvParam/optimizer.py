import logging
import numpy as np
from abc import abstractmethod
from typing import Dict, Union, Tuple
from timeit import default_timer as timer

from pymor.vectorarrays.interface import VectorArray

from RBInvParam.model import InstationaryModelIP
from RBInvParam.gradient_descent import gradient_descent_linearized_problem
from RBInvParam.reductor import InstationaryModelIPReductor
from RBInvParam.utils.logger import get_default_logger

MACHINE_EPS = 1e-16

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
        assert self.optimizer_parameter['noise_level'] >= 0
        assert self.optimizer_parameter['tau'] > 0
        assert self.optimizer_parameter['alpha_0'] >= 0
        assert self.optimizer_parameter['i_max'] >= 1

        if 'i_max_inner' in self.optimizer_parameter.keys():
            assert self.optimizer_parameter['i_max_inner'] >= 1

        #assert self.optimizer_parameter['reg_loop_max'] >= 1
        assert 0 < self.optimizer_parameter['theta'] \
                 < self.optimizer_parameter['Theta'] \
                 < 1
        
    
    def IRGNM(self,
              model: InstationaryModelIP,
              q_0: VectorArray,
              alpha_0: float, 
              tol : float,
              tau : float,
              noise_level : float,
              i_max : int,
              theta: float, 
              Theta : float,
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

        IRGNM_statistics['q'].append(q)
        IRGNM_statistics['J'].append(J)
        IRGNM_statistics['norm_nabla_J'].append(np.linalg.norm(nabla_J.to_numpy()))
        IRGNM_statistics['alpha'].append(alpha)

        self.logger.debug("Running IRGNM: ")
        self.logger.debug(f"  J : {J:3.4e}")
        self.logger.debug(f"  norm_nabla_J : {np.linalg.norm(nabla_J.to_numpy()):3.4e}")
        self.logger.debug(f"  alpha_0 : {alpha_0:3.4e}")
        self.logger.debug(f"  tol : {tol:3.4e}")
        self.logger.debug(f"  tau : {tau:3.4e}")
        self.logger.debug(f"  i_max : {i_max:3.4e}")
        self.logger.debug(f"  reg_loop_max : {reg_loop_max:3.4e}")
        self.logger.debug(f"  theta : {theta:3.4e}")
        self.logger.debug(f"  Theta : {Theta:3.4e}")

        while J >= tol+tau*noise_level and i<i_max:
            self.logger.info(f"##############################################################################################################################")
            self.logger.warning(f"IRGNM: Iteration {i} | J = {J:3.4e} is not sufficent: {J:3.4e} > {(tol+tau*noise_level):3.4e}.")
            self.logger.info(f'Start IRGNM iteration {i}: J = {J:3.4e}, norm_nabla_J = {np.linalg.norm(nabla_J.to_numpy()):3.4e}, alpha = {alpha:1.4e}')            
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

            while (not regularization_qualification) and (count < reg_loop_max):
                
                if not condition_low:
                    alpha *= 1.5  
                elif not condition_up:
                    alpha = alpha/2
                    #alpha = max(alpha/2,1e-14)
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

            if (count < reg_loop_max):
                self.logger.warning(f"Used alpha = {alpha:3.4e} does satisfy selection criteria: {theta*J:3.4e} < {2* lin_J:3.4e} < {Theta*J:3.4e}")
            else:
                self.logger.error(f"Not found valid alpha before reaching maximum number of tries : {reg_loop_max}.\n\
                                   Using the last alpha tested = {alpha:3.4e}.")
            q += d
            u = model.solve_state(q)
            p = model.solve_adjoint(q, u)
            J = model.objective(u)
            nabla_J = model.gradient(u, p)

            IRGNM_statistics['q'].append(q)
            IRGNM_statistics['J'].append(J)
            IRGNM_statistics['norm_nabla_J'].append(np.linalg.norm(nabla_J.to_numpy()))
            IRGNM_statistics['alpha'].append(alpha)
        
            #stagnation check
            if i > 3:
                buffer = IRGNM_statistics['J'][-3:]
                if abs(buffer[0] - buffer[1]) < MACHINE_EPS and abs(buffer[1] -buffer[2]) < MACHINE_EPS:
                    IRGNM_statistics['stagnation_flag'] = True
                    self.logger.info(f"Stop at iteration {i+1} of {int(i_max)}, due to stagnation.")
                    break

            IRGNM_statistics['time_steps'].append((timer()- start_time))
            self.logger.info(f'Statistics IRGNM iteration {i}: J = {J:3.4e}, norm_nabla_J = {np.linalg.norm(nabla_J.to_numpy()):3.4e}, alpha = {alpha:1.4e}')
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
                                        i_max = self.optimizer_parameter['i_max'],
                                        theta = self.optimizer_parameter['theta'],
                                        Theta = self.optimizer_parameter['Theta'],
                                        reg_loop_max = self.optimizer_parameter['reg_loop_max'])

        self.statistics['q'] = IRGNM_statistic['q']
        self.statistics['time_steps'] = IRGNM_statistic['time_steps']
        self.statistics['alpha'] = IRGNM_statistic['alpha']
        self.statistics['J'] = IRGNM_statistic['J']
        self.statistics['norm_nabla_J'] = IRGNM_statistic['norm_nabla_J']
        self.statistics['total_runtime'] = IRGNM_statistic['total_runtime']
        self.statistics['stagnation_flag'] = IRGNM_statistic['stagnation_flag']

        return q
        
class QrROMOptimizer(Optimizer):
    def __init__(self, 
                 optimizer_parameter: Dict, 
                 FOM : InstationaryModelIP,
                 logger: logging.Logger = None) -> None:

        super().__init__(optimizer_parameter, FOM, logger)
        self.reductor = InstationaryModelIPReductor(
            FOM
        )
        self.Qr_ROM = None

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
        alpha_0 = self.optimizer_parameter['alpha_0']
        tol = self.optimizer_parameter['tol']
        tau = self.optimizer_parameter['tau']
        i_max = self.optimizer_parameter['i_max']
        i_max_inner = self.optimizer_parameter['i_max_inner']
        reg_loop_max = self.optimizer_parameter['reg_loop_max']
        noise_level = self.optimizer_parameter['noise_level']
        theta = self.optimizer_parameter['theta']
        Theta = self.optimizer_parameter['Theta']

        start_time = timer()
        i = 0
        alpha = alpha_0
        delta = noise_level


        q = self.FOM.Q.make_array(self.optimizer_parameter['q_0'].copy())
        u = self.FOM.solve_state(q)
        p = self.FOM.solve_adjoint(q, u)
        J = self.FOM.objective(u)
        nabla_J = self.FOM.gradient(u, p)
        
        self.statistics['q'].append(q)
        self.statistics['alpha'].append(alpha)
        self.statistics['J'].append(J)
        self.statistics['norm_nabla_J'].append(np.linalg.norm(nabla_J.to_numpy()))


        self.logger.debug("Running Qr-IRGNM:")
        self.logger.debug(f"  J : {J:3.4e}")
        self.logger.debug(f"  norm_nabla_J : {np.linalg.norm(nabla_J.to_numpy()):3.4e}")
        self.logger.debug(f"  alpha_0 : {alpha_0:3.4e}")
        self.logger.debug(f"  tol : {tol:3.4e}")
        self.logger.debug(f"  tau : {tau:3.4e}")
        self.logger.debug(f"  i_max : {i_max:3.4e}")
        self.logger.debug(f"  i_max_inner : {i_max_inner:3.4e}")
        self.logger.debug(f"  reg_loop_max : {reg_loop_max:3.4e}")
        self.logger.debug(f"  noise_level : {noise_level:3.4e}")
        self.logger.debug(f"  theta : {theta:3.4e}")
        self.logger.debug(f"  Theta : {Theta:3.4e}")
        
        self.logger.debug(f"Extending Qr-space")
        q_basis_extention = self.FOM.Q.empty()
        q_basis_extention.append(nabla_J)
        q_basis_extention.append(q)
        q_basis_extention.append(self.FOM.Q.make_array(self.FOM.model_parameter['q_circ']))

        self.reductor.extend_basis(
             U = q_basis_extention,
             basis = 'parameter_basis'
        )
        self.Qr_ROM = self.reductor.reduce()
        self.logger.debug(f"Dim Qr-space = {self.reductor.get_dim('parameter_basis')}")
        self.logger.debug(f"Dim Vr-space = {self.reductor.get_dim('state_basis')}")

        while J >= tol+tau*noise_level and i<i_max:
            self.logger.info(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            self.logger.warning(f"Qr-IRGNM iteration {i}: J = {J:3.4e} is not sufficent: {J:3.4e} > {(tol+tau*noise_level):3.4e}.")
            self.logger.info(f'Start Qr-IRGNM iteration {i}: J = {J:3.4e}, norm_nabla_J = {np.linalg.norm(nabla_J.to_numpy()):3.4e}, alpha = {alpha:1.4e}')
            self.logger.info(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            q_r = self.reductor.project_vectorarray(q, 'parameter_basis')
            q_r = self.Qr_ROM.Q.make_array(q_r)

            q_r, IRGNM_statistic = self.IRGNM(model = self.Qr_ROM,
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
            self.statistics['norm_nabla_J'].append(np.linalg.norm(nabla_J.to_numpy()))

            if i > 3:
                buffer = self.statistics['J'][-3:]
                if abs(buffer[0] - buffer[1]) < MACHINE_EPS and abs(buffer[1] - buffer[2]) < MACHINE_EPS:
                    self.statistics['stagnation_flag'] = True
                    self.logger.info(f"Stop at iteration {i+1} of {int(i_max)}, due to stagnation.")
                    break
            
            self.logger.debug(f"Extending Qr-space")
            self.reductor.extend_basis(
                U = nabla_J,
                basis = 'parameter_basis'
            )
            self.Qr_ROM = self.reductor.reduce()
            self.logger.debug(f"Dim Qr-space = {self.reductor.get_dim('parameter_basis')}")
            self.logger.debug(f"Dim Vr-space = {self.reductor.get_dim('state_basis')}")

        self.statistics['total_runtime'] = (timer() - start_time)        
        return q
        

    def _initialize_optimization(self):    
        pass    

    def _get_regulatization(self):
        pass
    
    def _solve_subproblem(self):
        pass

    def _get_q_AGC(self):
        pass

    def _check_q_trial(self) -> bool:
        pass