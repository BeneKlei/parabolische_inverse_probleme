import logging
from abc import abstractmethod
from typing import Dict
from timeit import default_timer as timer
import numpy as np

from pymor.vectorarrays.interface import VectorArray

from model import InstationaryModelIP
from helper import gradient_descent

class Optimizer:
    def __init__(self, 
                 optimizer_parameter: Dict, 
                 FOM : InstationaryModelIP) -> None:
        self.FOM = FOM
        self.optimizer_parameter = optimizer_parameter
        self._check_optimizer_parameter()        
        self.logger = logging.getLogger(self.__class__.__name__)


        self.statistics = {
            'main_loop_idx': [],
            'q' : [],
            'time_steps' : [],
            'alpha' : [],
            'J' : [],
            'alpha_hist' : [],
            'total_runtime' : np.nan,
            'time_steps' : [],
            'stagnation_flag' : False,
            'optimizer_parameter' : self.optimizer_parameter.copy()
        }


    
    def _check_optimizer_parameter(self) -> None:
        assert self.optimizer_parameter['noise_level'] >= 0
        assert self.optimizer_parameter['tau'] > 0
        assert self.optimizer_parameter['alpha_0'] > 0
        assert self.optimizer_parameter['i_max'] >= 1
        assert self.optimizer_parameter['reg_loop_max'] >= 1
        assert 0 < self.optimizer_parameter['theta'] \
                 < self.optimizer_parameter['Theta'] \
                 < 1
    
    @abstractmethod
    def solve(self) -> VectorArray:
        pass

    def log_initial(self) -> None:
        #self.logger.debug("")
        pass

    def log_intermed_results(self) -> None:
        self.logger.debug("")
        

    def log_final_results(self) -> None:
        pass

    

class FOMOptimizer(Optimizer):
    def solve(self) -> VectorArray:
        start_time = timer()
        i = 0
        alpha = self.optimizer_parameter['alpha_0']
        q = self.optimizer_parameter['q_0'].copy()
    
        u = self.FOM.solve_state(q)
        J = self.FOM.objective(u)

        
        tol = self.optimizer_parameter['tol']
        tau = self.optimizer_parameter['tau']
        noise_level = self.optimizer_parameter['noise_level']
        i_max = self.optimizer_parameter['i_max']
        theta = self.optimizer_parameter['theta']
        Theta = self.optimizer_parameter['Theta']
        reg_loop_max = self.optimizer_parameter['reg_loop_max']

        while J >= tol+tau*noise_level and i<i_max:
            self.logger.info(f"\n ####################################################### \n")
            self.logger.info(f"Start main loop iteration i = {i}:")

            regularization_qualification = False
            count = 1#

            d = self.solve_linearized_problem()
            
            lin_u = self.FOM.solve_linearized_state(q, d, u)
            lin_p = self.FOM.solve_linearized_adjoint(q, u, lin_u)
            lin_J = self.FOM.linearized_objective(q, d, u, lin_u, alpha)

            condition_low = 0.5*theta*J<lin_J
            condition_up = lin_J < 0.5*Theta*J
            regularization_qualification = condition_low and condition_up


            while (not regularization_qualification) and (count < reg_loop_max) :
                self.logger.info(f"\t Searching for alpha:")
                if not condition_low:
                    alpha *= 1.5  
                elif not condition_up:
                    alpha = max(alpha/2,1e-14)
                else:
                    raise ValueError

                #d = ...
                d = q

                lin_u = self.FOM.solve_linearized_state(q, d, u)
                lin_p = self.FOM.solve_linearized_adjoint(q, u, lin_u)
                lin_J = self.FOM.linearized_objective(q, d, u, lin_u, alpha)

                condition_low = 0.5*theta*J<lin_J
                condition_up = lin_J < 0.5*Theta*J
                regularization_qualification = condition_low and condition_up

                self.logger.info(f"\t Test alpha = {alpha}.")
                self.logger.info(f"\t Try {count}: {0.5*theta*J:3.4e} < {lin_J:3.4e} < {0.5*Theta*J:3.4e}?")

                count += 1

            if (count < reg_loop_max):
                self.logger.info(f"ABC")
            else:
                self.logger.info(f"ABC")


            q += d
            u = self.FOM.solve_state(q)
            J = self.FOM.objective(u)        
        
            # stagnation check
            if i > 3:
                raise NotImplementedError

            i += 1
            self.statistics['time_steps'].append((timer()- start_time))

        
        self.statistics['total_runtime'] = (timer() - start_time)
        return q


    def solve_linearized_problem(self,
                                 q : np.array,
                                 d_start : np.array,
                                 method : str,
                                 kwargs : Dict) -> np.array:
        
        if method == 'gd':
            return gradient_descent(q, d_start, kwargs)
        
        elif method == 'cg':
            raise NotImplementedError
        else:
            raise ValueError



        



class ROMOptimizer(Optimizer):
    def __init__(self) -> None:
        self.FOM = None
        self.ROM = None
        self.reductor = None


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