import logging
import numpy as np
from abc import abstractmethod
from typing import Dict, Union, Tuple
from timeit import default_timer as timer

from pymor.vectorarrays.interface import VectorArray

from model import InstationaryModelIP
from gradient_descent import gradient_descent_linearized_problem
from reductor import InstationaryModelIPReductor

MACHINE_EPS = 1e-16

class Optimizer:
    def __init__(self, 
                 optimizer_parameter: Dict, 
                 FOM : InstationaryModelIP) -> None:
        self.FOM = FOM
        self.optimizer_parameter = optimizer_parameter
        self._check_optimizer_parameter()    
        logging.basicConfig()

        # TODO Add class and function logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.IRGNM_idx = 0


    
    def _check_optimizer_parameter(self) -> None:
        assert self.optimizer_parameter['noise_level'] >= 0
        assert self.optimizer_parameter['tau'] > 0
        assert self.optimizer_parameter['alpha_0'] >= 0
        assert self.optimizer_parameter['i_max'] >= 1
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
        assert noise_level > 0
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

        # TODO Own logger here
        # TODO Color logger for better readablity.

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


        while J >= tol+tau*noise_level and i<i_max:
            self.logger.info(f"###############################################################")
            self.logger.info(f"J = {J:3.4e} is still not sufficent; J > tol+tau*noise_level: {J:3.4e} > {(tol+tau*noise_level):3.4e}.")
            self.logger.info(f"Start main loop iteration i = {i}: J = {J:3.4e}.")

            regularization_qualification = False
            count = 1

            d_start = q.to_numpy().copy()
            d_start[:,:] = 0
            d_start = model.Q.make_array(d_start)

            max_iter = 1e4
            gc_tol = 1e-14
            inital_step_size = 1
            #TODO 
            d = self.solve_linearized_problem(model=model,
                                              q=q,
                                              d_start=d_start,
                                              alpha=alpha,
                                              method='gd',
                                              max_iter=max_iter,
                                              tol=gc_tol,
                                              inital_step_size=inital_step_size)
            
            lin_u = model.solve_linearized_state(q, d, u)
            lin_J = model.linearized_objective(q, d, u, lin_u, alpha=0)

            condition_low = theta*J< 2*lin_J
            condition_up = 2* lin_J < Theta*J
            regularization_qualification = condition_low and condition_up

            self.logger.info(f"alpha = {alpha} does not satisfy selection criteria.")
            self.logger.info(f"{theta*J:3.4e} < {2* lin_J:3.4e} < {Theta*J:3.4e}?")

            if (not regularization_qualification) and (count < reg_loop_max):
                self.logger.info(f"Searching for alpha:") 

            while (not regularization_qualification) and (count < reg_loop_max) :
                
                if not condition_low:
                    alpha *= 1.5  
                elif not condition_up:
                    alpha = alpha/2
                    #alpha = max(alpha/2,1e-14)
                else:
                    raise ValueError
                
                self.logger.info(f"Test alpha = {alpha}.")
                d = self.solve_linearized_problem(model=model,
                                                  q=q,
                                                  d_start=d_start,
                                                  alpha=alpha,
                                                  method='gd',
                                                  max_iter=max_iter,
                                                  tol=gc_tol,
                                                  inital_step_size=inital_step_size)

                lin_u = model.solve_linearized_state(q, d, u)
                lin_J = model.linearized_objective(q, d, u, lin_u, alpha=0)

                condition_low = theta*J< 2 * lin_J
                condition_up = 2* lin_J < Theta*J
                regularization_qualification = condition_low and condition_up
                            

                self.logger.info(f"Try {count}: {theta*J:3.4e} < {2* lin_J:3.4e} < {Theta*J:3.4e}?")
                self.logger.info(f"--------------------------------------------------------------------")

                count += 1

            if (count < reg_loop_max):
                self.logger.info(f"Found valid alpha = {alpha}.")
            else:
                self.logger.info(f"Not found valid alpha before reaching maximum number of tries :  {reg_loop_max}. \n \
                                   Using the last alpha tested = {alpha}.")
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
                    self.logger.info(f"Stop at iteration {i+1} of {int(max_iter)}, due to stagnation.")
                    break

            i += 1
            IRGNM_statistics['time_steps'].append((timer()- start_time))
            self.logger.info(f'i = {i}, J = {J:3.4e}, norm_nabla_J = {np.linalg.norm(nabla_J.to_numpy()):3.4e}, alpha = {alpha:1.4e}')
        
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
                 FOM : InstationaryModelIP) -> None:

        super().__init__(optimizer_parameter, FOM)
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
                 FOM : InstationaryModelIP) -> None:

        super().__init__(optimizer_parameter, FOM)
        self.reductor = InstationaryModelIPReductor(FOM)

    def solve(self):
        while not self._check_termination_criteria():
            pass

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