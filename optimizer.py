import logging
import numpy as np
from abc import abstractmethod
from typing import Dict, Union
from timeit import default_timer as timer
from pathlib import Path
from datetime import datetime

from pymor.vectorarrays.interface import VectorArray
from pymor.core.pickle import dump


from model import InstationaryModelIP
from gradient_descent import gradient_descent_linearized_problem

MACHINE_EPS = 1e-16

class Optimizer:
    def __init__(self, 
                 optimizer_parameter: Dict, 
                 FOM : InstationaryModelIP) -> None:
        self.FOM = FOM
        self.optimizer_parameter = optimizer_parameter
        self._check_optimizer_parameter()    
        logging.basicConfig()
        self.logger = logging.getLogger(self.__class__.__name__)


        self.statistics = {
            'q' : [],
            'time_steps' : [],
            'alpha' : [],
            'J' : [],
            'total_runtime' : np.nan,
            'stagnation_flag' : False,
            'optimizer_parameter' : self.optimizer_parameter.copy()
        }


    
    def _check_optimizer_parameter(self) -> None:
        assert self.optimizer_parameter['noise_level'] >= 0
        assert self.optimizer_parameter['tau'] > 0
        assert self.optimizer_parameter['alpha_0'] >= 0
        assert self.optimizer_parameter['i_max'] >= 1
        #assert self.optimizer_parameter['reg_loop_max'] >= 1
        assert 0 < self.optimizer_parameter['theta'] \
                 < self.optimizer_parameter['Theta'] \
                 < 1
    
    @abstractmethod
    def solve(self) -> VectorArray:
        pass

    def save_statistics(self,
                        path: Union[str, Path]) -> None:
        path = Path(path)
        assert path.suffix in ['.pkl', 'pickle']
        assert path.parent.exists()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_with_timestamp = f"{timestamp}_{path.stem}{path.suffix}"
        path_with_timestamp = path.parent / filename_with_timestamp
        
        with open(path_with_timestamp, 'wb') as file:
            dump(self.statistics, file)

    
class FOMOptimizer(Optimizer):
    def solve(self) -> VectorArray:
        start_time = timer()
        i = 0
        alpha = self.optimizer_parameter['alpha_0']
        q = self.optimizer_parameter['q_0'].copy()
        q = self.FOM.Q.make_array(q)
    
        u = self.FOM.solve_state(q)
        J = self.FOM.objective(u)
        
        tol = self.optimizer_parameter['tol']
        tau = self.optimizer_parameter['tau']
        noise_level = self.optimizer_parameter['noise_level']
        i_max = self.optimizer_parameter['i_max']
        theta = self.optimizer_parameter['theta']
        Theta = self.optimizer_parameter['Theta']
        reg_loop_max = self.optimizer_parameter['reg_loop_max']

        self.statistics['q'].append(q)
        self.statistics['J'].append(J)
        self.statistics['alpha'].append(alpha)

        # TODO Color logger for better readablity.

        while J >= tol+tau*noise_level and i<i_max:
            self.logger.info(f"########################################################")
            self.logger.info(f"J = {J:3.4e} is still not sufficent; J > tol+tau*noise_level: {J:3.4e} > {(tol+tau*noise_level):3.4e}.")
            self.logger.info(f"Start main loop iteration i = {i}: J = {J:3.4e}.")

            regularization_qualification = False
            count = 1

            d_start = q.to_numpy().copy()
            d_start[:,:] = 0
            d_start = self.FOM.Q.make_array(d_start)

            max_iter = 1e4
            gc_tol = 1e-14
            inital_step_size = 1
            #TODO 
            d = self.solve_linearized_problem(q=q,
                                              d_start=d_start,
                                              alpha=alpha,
                                              method='gd',
                                              max_iter=max_iter,
                                              tol=gc_tol,
                                              inital_step_size=inital_step_size)
            
            lin_u = self.FOM.solve_linearized_state(q, d, u)
            lin_J = self.FOM.linearized_objective(q, d, u, lin_u, alpha=0)

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
                    alpha = max(alpha/2,1e-14)
                else:
                    raise ValueError
                
                self.logger.info(f"Test alpha = {alpha}.")
                d = self.solve_linearized_problem(q=q,
                                                  d_start=d_start,
                                                  alpha=alpha,
                                                  method='gd',
                                                  max_iter=max_iter,
                                                  tol=gc_tol,
                                                  inital_step_size=inital_step_size)

                lin_u = self.FOM.solve_linearized_state(q, d, u)
                lin_J = self.FOM.linearized_objective(q, d, u, lin_u, alpha=0)

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
            u = self.FOM.solve_state(q)
            J = self.FOM.objective(u)        

            self.statistics['q'].append(q)
            self.statistics['J'].append(J)
            self.statistics['alpha'].append(alpha)
        
            #stagnation check
            if i > 3:
                buffer = self.statistics['J'][-3:]
                if abs(buffer[0] - buffer[1]) < MACHINE_EPS and abs(buffer[1] -buffer[2]) < MACHINE_EPS:
                    self.logger.info(f"Stop at iteration {i+1} of {int(max_iter)}, due to stagnation.")
                    break

            i += 1
            self.statistics['time_steps'].append((timer()- start_time))
            self.logger.info(f'i = {i}, J = {J:3.4e}, alpha = {alpha:1.4e}')
        
        self.statistics['total_runtime'] = (timer() - start_time)
        return q


    def solve_linearized_problem(self,
                                 q : np.array,
                                 d_start : np.array,
                                 alpha : float,
                                 method : str,
                                 **kwargs : Dict) -> np.array:
        
        if method == 'gd':
            return gradient_descent_linearized_problem(self.FOM, q, d_start, alpha, **kwargs)
        elif method == 'cg':
            raise NotImplementedError
        else:
            raise ValueError

class ROMOptimizer(Optimizer):
    def __init__(self, 
                 optimizer_parameter: Dict, 
                 FOM : InstationaryModelIP) -> None:

        super().__init__(optimizer_parameter, FOM)
        # self.reductor = InstationaryModelIPReductor(
            
        # )



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