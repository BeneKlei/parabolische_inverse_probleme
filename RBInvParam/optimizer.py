import logging
import numpy as np
import copy

from abc import abstractmethod
from typing import Dict, Union, Tuple, List
from timeit import default_timer as timer
from pathlib import Path

import pymor_dealii_bindings as pd2

from pymor.vectorarrays.interface import VectorArray
from pymor.algorithms.hapod import inc_vectorarray_hapod
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace
from pymor.operators.interface import Operator
from pymor.core.base import BasicObject
from pymor.core.exceptions import ExtensionError

from RBInvParam.model import InstationaryModelIP
from RBInvParam.linear_solver.gradient_descent import gradient_descent_linearized_problem
from RBInvParam.linear_solver.BiCGSTAB import BiCGStab_linearized_problem
from RBInvParam.reductor import InstationaryModelIPReductor
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.utils.io import save_dict_to_pkl, dealii_vector_space_to_numpy
from RBInvParam.domain_projector import SimpleBoundDomainProjector


MACHINE_EPS = 1e-16
STAGNATION_TOL = 1e-6

class Optimizer(BasicObject):
    def __init__(self, 
                 optimizer_parameter: Dict, 
                 FOM : InstationaryModelIP,
                 save_path: Path,
                 logger: logging.Logger = None) -> None:
                 
        self.FOM = FOM
        self.optimizer_parameter = optimizer_parameter
        self._check_optimizer_parameter()    
        logging.basicConfig()

        if logger:
            self._logger = logger
        else:
            self._logger = get_default_logger(self.__class__.__name__)
            self._logger.setLevel(logging.DEBUG)
        self.logger.debug(f"Setting up {self.__class__.__name__}")

        save_path = Path(save_path)
        assert save_path.exists()
        self.save_path = save_path

        self.name = None
        self.IRGNM_idx = 0
        self.IRGNM_statistics = {}

        self.linear_solver_operator = None

    def _check_optimizer_parameter(self) -> None:
        keys = self.optimizer_parameter.keys()

        assert self.optimizer_parameter["alpha_0"] >= 0
        assert self.optimizer_parameter["tol"] > 0
        assert self.optimizer_parameter["tau"] > 0
        assert self.optimizer_parameter["noise_level"] >= 0
        assert 0 < self.optimizer_parameter["theta"] \
                 < self.optimizer_parameter["Theta"] 
                 #\ < 1
        if "tau_tilde" in keys:
            assert self.optimizer_parameter["tau_tilde"] > 0

        assert self.optimizer_parameter["i_max"] >= 1
        if "i_max_inner" in keys:
            assert self.optimizer_parameter["i_max_inner"] >= 1
        assert self.optimizer_parameter["reg_loop_max"] >= 1
        if "armijo_max_iter" in keys:
            assert self.optimizer_parameter["armijo_max_iter"] >= 1

        if "eta0" in keys:
            assert self.optimizer_parameter["eta0"] > 0
        if "kappa_arm" in keys:
            assert self.optimizer_parameter["kappa_arm"] > 0
        if "beta_1" in keys:
            assert 0 < self.optimizer_parameter["beta_1"] < 1
        if "beta_2" in keys:
            assert 3/4 <= self.optimizer_parameter["beta_2"] < 1
        if "beta_3" in keys:
            assert 0 < self.optimizer_parameter["beta_3"] < 1    
    
    def _armijo_TR_line_serach(self,
                               model: InstationaryModelIP, 
                               previous_q: NumpyVectorArray,
                               previous_J: float,
                               search_direction : NumpyVectorArray,
                               max_iter: int,
                               inital_step_size: float,
                               eta: float,
                               beta: float,
                               kappa_arm: float,
                               use_cached_operators: bool = False,
                               projector: SimpleBoundDomainProjector = None) -> Tuple[NumpyVectorArray, float, bool]:

        assert 0 <= beta < 1
        assert 0 < eta
        
        i = 0
        model_unsufficent = False
        TR_max_iter_cond = False

        self.logger.info(f"Start Armijo backtracking, with J = {previous_J:3.4e}.")
        step_size = inital_step_size
        #search_direction.scal(1.0 / model.compute_gradient_norm(search_direction))

        if projector:
            projector.pre_compute(center=previous_q)
            current_q = projector.project_domain(previous_q, step_size * search_direction)
        else:
            current_q = previous_q + step_size * search_direction

        u = model.solve_state(q=current_q, use_cached_operators=use_cached_operators)
        p = model.solve_adjoint(q=current_q, u=u, use_cached_operators=use_cached_operators)
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
        if current_J > 0:
            abs_est_error_J_r, _ = self.estimate_objective_error(
                model = model,
                q = current_q,
                u = u,
                p = p,
                use_cached_operators=use_cached_operators
            )
            J_rel_error = abs_est_error_J_r / current_J
        else:
            J_rel_error = np.inf
        
        print("--------")
        print(J_rel_error)
        
        TR_condition = J_rel_error <= eta
        condition = armijo_condition & TR_condition
        i += 1

        print("############")
        print(lhs)
        print(armijo_condition)
        print(TR_condition)
        
        while (not condition) and (i < max_iter):
            step_size = 0.5 * step_size
            
            if projector: 
                projector.pre_compute(center=previous_q)
                current_q = projector.project_domain(previous_q, step_size * search_direction)
            else:
                current_q = previous_q + step_size * search_direction

            u = model.solve_state(q=current_q, use_cached_operators=use_cached_operators)
            p = model.solve_adjoint(q=current_q, u=u, use_cached_operators=use_cached_operators)
            current_J = model.objective(u=u,
                                        q=current_q)
            
            norm_d = model.compute_gradient_norm(previous_q - current_q)
            lhs = previous_J - current_J
            rhs = kappa_arm / step_size * norm_d**2

            # print("A")
            # print(previous_J)
            # print(current_J)
            # print(lhs)
            # print(rhs)
            
            if abs(lhs) <= MACHINE_EPS:
                lhs = 0

            if abs(rhs) <= MACHINE_EPS:
                rhs = 0

            # print("############")
            # print(lhs)
            # print(rhs)
            # print(step_size)

            armijo_condition = lhs >= rhs

            if current_J > 0:
                abs_est_error_J_r, _ = self.estimate_objective_error(
                    model = model,
                    q = current_q,
                    u = u,
                    p = p,
                    use_cached_operators=use_cached_operators
                )                
                J_rel_error = abs_est_error_J_r / current_J
            else:
                J_rel_error = np.inf

            TR_condition = J_rel_error <= eta
            condition = armijo_condition & TR_condition

            print("############")
            print(lhs)
            print(armijo_condition)
            print(TR_condition)
            # print(step_size)

            
            i += 1
        if (J_rel_error > beta * eta):
            model_unsufficent = True
        
        if i == max_iter:
            TR_max_iter_cond = True
    

        if not condition:
            self.logger.error(f"Armijo backtracking does NOT terminate normally. step_size = {step_size:3.4e}; Stopping at J = {current_J:3.4e}")
            self.logger.debug(f"armijo_condition = {armijo_condition}, TR_condition = {TR_condition}")

        else:
            self.logger.debug(f"Armijo backtracking does terminate normally with step_size = {step_size:3.4e}; Stopping at J = {current_J:3.4e}")

        return (current_q, current_J, model_unsufficent, TR_max_iter_cond, step_size)
    
    def estimate_objective_error(self,
                                 model: InstationaryModelIP,
                                 q : VectorArray,
                                 u : VectorArray,
                                 p : VectorArray,
                                 use_cached_operators: bool = True) -> Tuple[float,float]:

        if id(model) == id(self.FOM):
            return 0.0, 0.0
        
        if not model.objective_error_estimator:
            print("Here")
            J = self.FOM.compute_objective(
                q = self.reductor.reconstruct(q, basis='parameter_basis')
            )
            J_r = model.compute_objective(
                q = q
            )

            # nabla_J = self.FOM.compute_gradient(
            #     q = self.reductor.reconstruct(q, basis='parameter_basis')
            # )
            # nabla_J_r = self.reductor.reconstruct(model.compute_gradient(
            #     q = q
            # ), basis='parameter_basis')

            # u = self.FOM.solve_state(q=self.reductor.reconstruct(q, basis='parameter_basis'), 
            #                  use_cached_operators=use_cached_operators)
            # p = self.FOM.solve_adjoint(q=self.reductor.reconstruct(q, basis='parameter_basis'), 
            #                     u=u, 
            #                     use_cached_operators=use_cached_operators)
            
            # u_r = model.solve_state(q=q, use_cached_operators=use_cached_operators)
            # p_r = model.solve_adjoint(q=q, 
            #                           u=u_r, 
            #                           use_cached_operators=use_cached_operators)
            
            # print(np.sqrt(self.FOM.products['bochner_prod_V'].apply2(p_r-p,p_r-p))[0,0] / np.sqrt(self.FOM.products['bochner_prod_V'].apply2(p,p))[0,0])
            

            # # print(J)
            # # print(J_r)
            # # print(nabla_J)
            # # print(nabla_J_r)
            # print(np.abs(J - J_r) / np.abs(J))
            # print("------")
            # print((nabla_J - nabla_J_r).to_numpy())
            # print(nabla_J.to_numpy())

            # # print((nabla_J - nabla_J_r).to_numpy() / nabla_J.to_numpy())
            # print(self.FOM.compute_gradient_norm(nabla_J - nabla_J_r) / self.FOM.compute_gradient_norm(nabla_J))
            
            #print(u - u_r)

            # print(r"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            # np.set_printoptions(threshold=np.inf)
            # print(u_r[-1])
            # print(p_r[0])

            return np.abs(J - J_r), np.nan
            #self.FOM.compute_gradient_norm(nabla_J - nabla_J_r)
            
        
        return self.model.estimate_objective_error(
            q = q,
            u = u,
            p = p,
            use_cached_operators=use_cached_operators
        )

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
              reg_loop_max: int,
              use_TR: bool = False,
              lin_solver_parms: Dict = None,
              TR_backtracking_params: Dict = None,
              use_cached_operators: bool = False,
              dump_IRGNM_intermed_stats: bool = False,
              dump_every_nth_loop: int = 0,
              projector: SimpleBoundDomainProjector = None) -> Tuple[VectorArray, Dict]: 

        assert q_0 in model.Q
        assert tol > 0
        assert tau > 0
        assert noise_level >= 0
        #assert 0 < theta < Theta < 1

        assert lin_solver_parms is not None        

        if use_TR:
            assert TR_backtracking_params is not None
            method_name = 'TR-IRGNM'
        else:
            method_name = 'IRGNM'

        stagnation_flag = False
        self.IRGNM_statistics = {
            'IRGNM_idx' : self.IRGNM_idx,
            "q" : [],
            'time_steps' : [],
            "alpha" : [],
            "J" : [],
            "norm_nabla_J" : [],
            "total_runtime" : [],
            "stagnation_flag" : False,
            "FOM_num_calls" : {},
            "counts" : None
        }
        counts = {
            'IRGNM_loop_iter' : -1,
            'reg_loop_iter' : [],
            'lin_solver_iter' : [],
            'loop_terminated' : []
        }

        start_time = timer()
        i = 0
        model_unsufficent = False
        
        # if not q_0 in self.FOM.Q:
        #     q_ = self.reductor.reconstruct(q_0, basis='parameter_basis')
        # else:
        #     q_ = q_0.copy()

        # q_ = q_.to_numpy().flatten()
        # mask_lb = q_ >= self.FOM.bounds[:,0]
        # mask_ub = q_ <= self.FOM.bounds[:,1]
        # assert np.all(mask_lb) and np.all(mask_ub)

        alpha = alpha_0
        q = q_0.copy()
        u = model.solve_state(q, use_cached_operators=use_cached_operators)
        p = model.solve_adjoint(q, u, use_cached_operators=use_cached_operators)
        J = model.objective(u)
        nabla_J = model.gradient(u, p, q, use_cached_operators=use_cached_operators)
        norm_nabla_J = model.compute_gradient_norm(nabla_J)

        self.IRGNM_statistics["q"].append(q)
        self.IRGNM_statistics["J"].append(J)
        self.IRGNM_statistics["norm_nabla_J"].append(norm_nabla_J)
        self.IRGNM_statistics["alpha"].append(alpha)

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


        while np.sqrt(2 * J) >= tol+tau*noise_level and i<i_max:
            self.logger.info(f"##############################################################################################################################")
            self.logger.warning(f"{method_name}: Iteration {i} | J = {J:3.4e} is not sufficent: {np.sqrt(2 * J):3.4e} > {(tol+tau*noise_level):3.4e}.")
            self.logger.info(f'Start {method_name} iteration {i}: J = {J:3.4e}, norm_nabla_J = {model.compute_gradient_norm(nabla_J):3.4e}, alpha = {alpha:1.4e}')
            self.logger.info(f"------------------------------------------------------------------------------------------------------------------------------")
            self.logger.info(f"Try 1: test alpha = {alpha:3.4e}.")

            regularization_qualification = False
            count = 1

            if projector:
                projector.pre_compute(center=q)

            d_start = q.to_numpy().copy()
            d_start[:,:] = 0
            d_start = model.Q.make_array(d_start)

            d, lin_solver_iter = self.solve_linearized_problem(model=model,
                                                               q=q,
                                                               d_start=d_start,
                                                               alpha=alpha,
                                                               lin_solver_parms = lin_solver_parms, 
                                                               logger = self.logger,
                                                               use_cached_operators=use_cached_operators,
                                                               projector=projector)

            counts['lin_solver_iter'].append([lin_solver_iter])
            
            lin_u = model.solve_linearized_state(q, d, u, use_cached_operators=use_cached_operators)
            lin_J = model.linearized_objective(q, d, u, lin_u, alpha=0, use_cached_operators=use_cached_operators)

            condition_low = theta*J< 2*lin_J
            condition_up = 2* lin_J < Theta*J
            regularization_qualification = condition_low and condition_up

            if (not regularization_qualification) and (count < reg_loop_max):
                self.logger.warning(f"Used alpha = {alpha:3.4e} does NOT satisfy selection criteria: {theta*J:3.4e} < {2* lin_J:3.4e} < {Theta*J:3.4e}")
                self.logger.info(f"Searching for alpha:") 

            loop_terminated = False
            while (not regularization_qualification) and (count < reg_loop_max):
                count += 1

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

                d, lin_solver_iter = self.solve_linearized_problem(model=model,
                                                                   q=q,
                                                                   d_start=d_start,
                                                                   alpha=alpha,
                                                                   lin_solver_parms = lin_solver_parms,
                                                                   logger = self.logger,
                                                                   use_cached_operators=use_cached_operators,
                                                                   projector=projector)
                
                counts['lin_solver_iter'][-1].append(lin_solver_iter)

                lin_u = model.solve_linearized_state(q, d, u, use_cached_operators=use_cached_operators)
                lin_J = model.linearized_objective(q, d, u, lin_u, alpha=0, use_cached_operators=use_cached_operators)

                condition_low = theta*J< 2 * lin_J
                condition_up = 2* lin_J < Theta*J
                regularization_qualification = condition_low and condition_up
                            
                if (not regularization_qualification) and (count < reg_loop_max):
                    self.logger.warning(f"Used alpha = {alpha:3.4e} does NOT satisfy selection criteria: {theta*J:3.4e} < {2* lin_J:3.4e} < {Theta*J:3.4e}")
                else:
                    self.logger.info(f"------------------------------------------------------------------------------------------------------------------------------")

            # print(".........................")
            # print(q)
            # print(d)
            # print(q+d)

            loop_terminated = loop_terminated or (count >= reg_loop_max)

            counts['reg_loop_iter'].append(count)
            counts['loop_terminated'].append(loop_terminated)

            if not loop_terminated:
                self.logger.warning(f"Used alpha = {alpha:3.4e} does satisfy selection criteria: {theta*J:3.4e} < {2* lin_J:3.4e} < {Theta*J:3.4e}")
            else:
                self.logger.error(f"Not found valid alpha before reaching maximum number of tries : {reg_loop_max}.\n\
                                   Using the last alpha tested = {alpha:3.4e}.")
                
                break
                
            ########################################### Armijo ###########################################
            if use_TR:
                self.logger.info(f"Enforcing TR condition.")
                q_TR, _, model_unsufficent, TR_max_iter_cond, step_size = self._armijo_TR_line_serach(
                    model = model,
                    previous_q = q,
                    previous_J = J,
                    search_direction = d,
                    **TR_backtracking_params,
                    use_cached_operators=use_cached_operators,
                    projector=projector
                )

                if TR_max_iter_cond:
                    break
                
                # print("|q-q_TR|")
                # print(model.compute_gradient_norm(q - q_TR))
                q = q_TR
            else:
                # print("|d|")
                # print(model.compute_gradient_norm(d))
                q += d

            ########################################### Final ###########################################

            u = model.solve_state(q, use_cached_operators=use_cached_operators)
            p = model.solve_adjoint(q, u, use_cached_operators=use_cached_operators)
            J = model.objective(u)
            nabla_J = model.gradient(u, p, q, use_cached_operators=use_cached_operators)
            norm_nabla_J = model.compute_gradient_norm(nabla_J)

            self.IRGNM_statistics["q"].append(q)
            self.IRGNM_statistics["J"].append(J)
            self.IRGNM_statistics["norm_nabla_J"].append(norm_nabla_J)
            self.IRGNM_statistics["alpha"].append(alpha)

            #stagnation check
            if i > 3:
                buffer = self.IRGNM_statistics["J"][-3:]
                if abs(buffer[0] - buffer[1]) / abs(buffer[0]) < STAGNATION_TOL and abs(buffer[1] - buffer[2]) / abs(buffer[1])< STAGNATION_TOL:
                    self.IRGNM_statistics["stagnation_flag"] = True
                    self.logger.info(f"Stop at iteration {i+1} of {int(i_max)}, due to stagnation.")
                    stagnation_flag = True
                    break

            self.IRGNM_statistics['time_steps'].append((timer()- start_time))
            self.logger.info(f'Statistics {method_name} iteration {i}: J = {J:3.4e}, norm_nabla_J = {norm_nabla_J:3.4e}, alpha = {alpha:1.4e}')
            i += 1
            if not(np.sqrt(2 * J) >= tol+tau*noise_level and i<i_max):
                self.logger.info(f"##############################################################################################################################")

            self.IRGNM_statistics["FOM_num_calls"] = self.FOM.num_calls
            if dump_IRGNM_intermed_stats:
                if self.name is not None:
                    save_path = self.save_path / f'{self.name}_IRGNM_{i}.pkl'
                else:
                    save_path = self.save_path / f'IRGNM_{i}.pkl'
                
                if (i % dump_every_nth_loop == 0) or (i == 1):
                    self.dump_stats(data=self.IRGNM_statistics,
                                    save_path=save_path)

            self.IRGNM_statistics["total_runtime"].append(timer() - start_time) 

            if model_unsufficent:
                break

        self.logger.info(f'Final {method_name} Statistics:')
        if loop_terminated:
            self.logger.info(f'     {method_name} No sufficient regularization constant found i = {i}')
        elif i == i_max and not model_unsufficent:
            self.logger.info(f'     {method_name} reached maxit at i = {i}')
        elif i < i_max and not model_unsufficent:
            self.logger.info(f'     {method_name} converged at i = {i}')
        elif TR_max_iter_cond:
            self.logger.info(f'     {method_name} TR backtracking reach maximum iteration number at i = {i}')
        elif model_unsufficent:
            self.logger.info(f'     {method_name} TR boundary criterium triggered at i = {i}')
        elif stagnation_flag:
            self.logger.info(f'     {method_name} TR stagnated at i = {i}')
        else:
            # Should never be happend
            raise NotImplementedError
                
        self.logger.info(f'     Start J = {self.IRGNM_statistics["J"][0]:3.4e}; Final J = {self.IRGNM_statistics["J"][-1]:3.4e}.')
        self.logger.info(f'     Start alpha = {self.IRGNM_statistics["alpha"][0]:3.4e}; Final alpha = {self.IRGNM_statistics["alpha"][-1]:3.4e}.')
        self.logger.info(f'     Start norm_nabla_J = {self.IRGNM_statistics["norm_nabla_J"][0]:3.4e}; Final norm_nabla_J = {self.IRGNM_statistics["norm_nabla_J"][-1]:3.4e}.')
        self.logger.info(f'     Euclidian distance final q and inital q = {np.linalg.norm(q.to_numpy() - q_0.to_numpy()):3.4e}')

        counts['IRGNM_loop_iter'] = self.IRGNM_idx

        self.IRGNM_statistics["counts"] = counts
        self.IRGNM_statistics["total_runtime"].append(timer() - start_time)
        self.IRGNM_idx += 1
        return (q, self.IRGNM_statistics)

    def solve_linearized_problem(self,
                                model : InstationaryModelIP, 
                                q : VectorArray,
                                d_start : VectorArray,
                                alpha : float,
                                use_cached_operators: bool,
                                logger: logging.Logger,
                                lin_solver_parms : Dict,
                                projector: SimpleBoundDomainProjector = None) -> Tuple[VectorArray, int]:

        method = lin_solver_parms['method']
        if method == 'gd':
            return gradient_descent_linearized_problem(model=model,
                                                       q = q, 
                                                       d_start = d_start, 
                                                       alpha = alpha,
                                                       use_cached_operators=use_cached_operators,
                                                       logger=logger,
                                                       lin_solver_parms = lin_solver_parms,
                                                       projector=projector)
        elif method == 'BiCGSTAB':
            return BiCGStab_linearized_problem(model, 
                                               q = q, 
                                               d_start = d_start, 
                                               alpha = alpha,
                                               use_cached_operators=use_cached_operators,
                                               logger=logger,
                                               lin_solver_parms = lin_solver_parms)


           
        else:
            raise ValueError
        
    def dump_stats(self, 
                   data: Dict,
                   save_path: Union[str, Path] = None):
        if not save_path:
            save_path = self.save_path
        
        save_path = Path(save_path)
        assert save_path.suffix in ['.pkl', 'pickle']
        assert save_path.parent.exists()

        self.logger.info(f"Dumping statistics IRGNM to {save_path}.")
        save_dict_to_pkl(path=save_path, data=data, use_timestamp=False)
    
    def dump_prepare_statistics(self, statistics: Dict) -> Dict:
        assert 'reduced_bases' in statistics.keys()
        _state_basis = statistics['reduced_bases']['state_basis']

        if isinstance(_state_basis.vectors[0].real_part.impl, pd2.Vector):
            statistics['reduced_bases']['state_basis'] = dealii_vector_space_to_numpy(statistics['reduced_bases']['state_basis'])
                
        return statistics
          
class FOMOptimizer(Optimizer):
    def __init__(self, 
                 optimizer_parameter: Dict, 
                 FOM : InstationaryModelIP,
                 save_path : Path,
                 logger: logging.Logger = None)-> None:

        super().__init__(optimizer_parameter = optimizer_parameter, 
                         FOM = FOM, 
                         logger = logger, 
                         save_path = save_path)
        
        self.statistics = {
            "q" : [],
            'time_steps' : [],
            "alpha" : [],
            "J" : [],
            "norm_nabla_J" : [],
            "total_runtime" : [],
            "stagnation_flag" : False,
            "optimizer_parameter" : self.optimizer_parameter.copy(),
            "FOM_num_calls" : {},
            "counts" : {}
        }

    def solve(self) -> VectorArray:
        q_0 = self.optimizer_parameter["q_0"].copy()
        alpha_0 = self.optimizer_parameter["alpha_0"]
        tol = self.optimizer_parameter["tol"]
        tau = self.optimizer_parameter["tau"]
        noise_level = self.optimizer_parameter["noise_level"]
        theta = self.optimizer_parameter["theta"]
        Theta = self.optimizer_parameter["Theta"]

        i_max = self.optimizer_parameter["i_max"]
        reg_loop_max = self.optimizer_parameter["reg_loop_max"]

        lin_solver_parms = self.optimizer_parameter['lin_solver_parms']
        use_cached_operators = self.optimizer_parameter['use_cached_operators']

        dump_every_nth_loop = self.optimizer_parameter['dump_every_nth_loop']

        q = self.FOM.Q.make_array(q_0)
        u = self.FOM.solve_state(q, use_cached_operators=use_cached_operators)
        p = self.FOM.solve_adjoint(q, u, use_cached_operators=use_cached_operators)
        J = self.FOM.objective(u)
        nabla_J = self.FOM.gradient(u, p, q, use_cached_operators=use_cached_operators)
        norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)

        self.logger.debug("Running FOM-IRGNM:")
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
        self.logger.debug(f"  reg_loop_max : {reg_loop_max:3.4e}")
        self.logger.debug(f"  lin_solver_parms : ")
        for (key,val) in lin_solver_parms.items():
            self.logger.debug(f"        {key} : {val}")
        self.logger.debug(f"  use_cached_operators : {use_cached_operators}")

        projector = SimpleBoundDomainProjector(
            model = self.FOM,
            bounds = self.FOM.bounds,
            reductor = None,
            use_sufficient_condition = False,
            logger = self.logger
        )

        self.name = 'FOM'
        q, IRGNM_statistic = self.IRGNM(model = self.FOM,
                                        q_0 = q,
                                        alpha_0 = alpha_0,
                                        tol = tol,
                                        tau = tau,
                                        noise_level = noise_level,
                                        theta = theta,
                                        Theta = Theta,
                                        i_max = i_max,
                                        reg_loop_max = reg_loop_max,
                                        lin_solver_parms = lin_solver_parms,
                                        use_cached_operators = use_cached_operators,
                                        dump_IRGNM_intermed_stats = True,
                                        dump_every_nth_loop=dump_every_nth_loop,
                                        projector=projector)

        self.statistics["q"] = IRGNM_statistic["q"]
        self.statistics['time_steps'] = IRGNM_statistic['time_steps']
        self.statistics["alpha"] = IRGNM_statistic["alpha"]
        self.statistics["J"] = IRGNM_statistic["J"]
        self.statistics["norm_nabla_J"] = IRGNM_statistic["norm_nabla_J"]
        self.statistics["total_runtime"] = IRGNM_statistic["total_runtime"]
        self.statistics["stagnation_flag"] = IRGNM_statistic["stagnation_flag"]
        self.statistics["FOM_num_calls"] = IRGNM_statistic["FOM_num_calls"]
        self.statistics["counts"] = IRGNM_statistic["counts"]

        self.dump_stats(data=self.statistics,
                        save_path = self.save_path / f'FOM_IRGNM_final.pkl')

        return q
        
class QrFOMOptimizer(Optimizer):
    def __init__(self, 
                 optimizer_parameter: Dict, 
                 FOM : InstationaryModelIP,
                 save_path : Path,
                 logger: logging.Logger = None) -> None:
        raise Exception("QrFOMOptimizer is deprecated.")
        super().__init__(optimizer_parameter = optimizer_parameter, 
                         FOM = FOM, 
                         logger = logger, 
                         save_path = save_path)

        self.reductor = InstationaryModelIPReductor(
            FOM
        )
        self.QrFOM = None

        self.statistics = {
            "q" : [],
            "alpha" : [],
            "J" : [],
            "norm_nabla_J" : [],
            "total_runtime" : np.nan,
            #'inner_loop_time_steps' : [],
            "stagnation_flag" : False,
            "optimizer_parameter" : self.optimizer_parameter.copy(),
            "FOM_num_calls" : {}
        }
    

    def solve(self) -> VectorArray:
        q_0 = self.optimizer_parameter["q_0"].copy()
        alpha_0 = self.optimizer_parameter["alpha_0"]
        tol = self.optimizer_parameter["tol"]
        tau = self.optimizer_parameter["tau"]
        noise_level = self.optimizer_parameter["noise_level"]
        theta = self.optimizer_parameter["theta"]
        Theta = self.optimizer_parameter["Theta"]

        i_max = self.optimizer_parameter["i_max"]
        reg_loop_max = self.optimizer_parameter["reg_loop_max"]
        i_max_inner = self.optimizer_parameter["i_max_inner"]

        lin_solver_parms = self.optimizer_parameter['lin_solver_parms']
        use_cached_operators = self.optimizer_parameter['use_cached_operators']

        start_time = timer()
        i = 0
        alpha = alpha_0
        delta = noise_level

        q = self.FOM.Q.make_array(q_0)
        u = self.FOM.solve_state(q, use_cached_operators=use_cached_operators)
        p = self.FOM.solve_adjoint(q, u, use_cached_operators=use_cached_operators)
        J = self.FOM.objective(u)
        nabla_J = self.FOM.gradient(u, p, q, use_cached_operators=use_cached_operators)
        norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)
        
        self.statistics["q"].append(q)
        self.statistics["alpha"].append(alpha)
        self.statistics["J"].append(J)
        self.statistics["norm_nabla_J"].append(norm_nabla_J)


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
        self.logger.debug(f"  lin_solver_parms : ")
        for (key,val) in lin_solver_parms.items():
            self.logger.debug(f"        {key} : {val}")
        self.logger.debug(f"  use_cached_operators : {use_cached_operators}")
        
        self.logger.debug(f"Extending Qr-space")
        self.parameter_shapshots = self.FOM.Q.empty()
        self.parameter_shapshots.append(nabla_J)
        self.parameter_shapshots.append(q)
        self.parameter_shapshots.append(self.FOM.Q.make_array(self.FOM.setup['q_circ']))

        if self.FOM.setup['q_time_dep']:
            self.logger.debug(f"Performing HaPOD on parameter snapshots.")
            _parameter_shapshots, _ = self._HaPOD(shapshots=self.parameter_shapshots, 
                                                  basis='parameter_basis',
                                                  product=self.FOM.products['prod_Q'])

        self.reductor.extend_basis(
             U = _parameter_shapshots,
             basis = 'parameter_basis'
        )
        self.QrFOM = self.reductor.reduce()


        while np.sqrt(2 * J) >= tol+tau*noise_level and i<i_max:
            self.logger.info(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            self.logger.warning(f"Qr-IRGNM iteration {i}: J = {J:3.4e} is not sufficent: {np.sqrt(2 * J):3.4e} > {(tol+tau*noise_level):3.4e}.")
            self.logger.info(f'Start Qr-IRGNM iteration {i}: J = {J:3.4e}, norm_nabla_J = {norm_nabla_J:3.4e}, alpha = {alpha:1.4e}')
            self.logger.info(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            q_r = self.reductor.project_vectorarray(q, 'parameter_basis')
            q_r = self.QrFOM.Q.make_array(q_r)
            
            self.IRGNM_idx += 1
            
            inner_loop_start_time = timer()
            q_r, IRGNM_statistic = self.IRGNM(model = self.QrFOM,
                                              q_0 = q_r,
                                              alpha_0 = alpha,
                                              tol = tol,
                                              tau = tau,
                                              noise_level = delta,
                                              i_max = i_max_inner,
                                              theta = theta,
                                              Theta = Theta,
                                              reg_loop_max = reg_loop_max,
                                              lin_solver_parms = lin_solver_parms,
                                              use_cached_operators = use_cached_operators)
            
            q = self.reductor.reconstruct(q_r, basis='parameter_basis')
            u = self.FOM.solve_state(q, use_cached_operators=use_cached_operators)
            p = self.FOM.solve_adjoint(q, u, use_cached_operators=use_cached_operators)
            J = self.FOM.objective(u)
            nabla_J = self.FOM.gradient(u, p, q, use_cached_operators=use_cached_operators)
            alpha = IRGNM_statistic["alpha"][1]

            self.statistics["q"].append(q)
            self.statistics["alpha"].append(alpha)
            self.statistics["J"].append(J)
            self.statistics["norm_nabla_J"].append(self.FOM.compute_gradient_norm(nabla_J))

            if i > 3:
                buffer = self.statistics["J"][-3:]
                if abs(buffer[0] - buffer[1]) < MACHINE_EPS and abs(buffer[1] - buffer[2]) < MACHINE_EPS:
                    self.statistics["stagnation_flag"] = True
                    self.logger.info(f"Stop at iteration {i+1} of {int(i_max)}, due to stagnation.")
                    break
            
            self.statistics["FOM_num_calls"] = self.FOM.num_calls
            self.dump_stats(data=self.statistics,
                            save_path = self.save_path / f'QrFOM_IRGNM_{i}.pkl')
            
            self.logger.debug(f"Extending Qr-space")
            self.parameter_shapshots = self.FOM.Q.empty()
            self.parameter_shapshots.append(nabla_J)
            
            if self.FOM.q_time_dep:
                self.logger.debug(f"Performing HaPOD on parameter snapshots.")
                _parameter_shapshots, _ = self._HaPOD(shapshots=self.parameter_shapshots, 
                                                      basis='parameter_basis',
                                                      product=self.FOM.products['prod_Q'])

            self.reductor.extend_basis(
                U = _parameter_shapshots,
                basis = 'parameter_basis'
            )
            self.QrFOM = self.reductor.reduce()
            self.logger.debug(f"Dim Qr-space = {self.reductor.get_bases_dim('parameter_basis')}")
            self.logger.debug(f"Dim Vr-space = {self.reductor.get_bases_dim('state_basis')}")

        self.statistics["total_runtime"] = (timer() - start_time)
        self.statistics["FOM_num_calls"] = self.FOM.num_calls
        self.dump_stats(data=self.statistics,
                        save_path = self.save_path / f'QrFOM_IRGNM_final.pkl')
        return q
        
class QrVrROMOptimizer(Optimizer):
    def __init__(self, 
                 optimizer_parameter: Dict, 
                 FOM : InstationaryModelIP,
                 save_path: Path,
                 logger: logging.Logger = None) -> None:

        super().__init__(optimizer_parameter = optimizer_parameter, 
                         FOM = FOM, 
                         logger = logger, 
                         save_path = save_path)

        self.reductor = InstationaryModelIPReductor(
            FOM,
            optimizer_parameter['error_estimator_types']
        )
        self.QrVrROM = None

        self.parameter_shapshots = None
        self.state_shapshots = None

        self.statistics = {
            "q" : [],
            "alpha" : [],
            "J" : [],
            "norm_nabla_J" : [],
            "J_r" : [],
            "abs_est_error_J_r" : [],
            "rel_est_error_J_r" : [],
            "abs_est_error_nabla_J_r" : [],
            "rel_est_error_nabla_J_r" : [],
            "total_runtime" : [],
            "outer_loop_runtime" : {
                'AGC_runtime' : [],
                'IRGNM_runtime' : [],
                'solve_snapshot_FOM_runtime' : [],
                'preprocess_parameter_snapshot_runtime' : [],
                'preprocess_state_snapshot_runtime' : [],
                'extend_parameter_basis_runtime' : [],
                'extend_state_basis_runtime' : [],
                'reduce_runtime' : [],
                'total_runtime' : []
            },
            "flags" : {
                "proj_q_in_tr" : [],
                "AGC_decay_cond" : [],
                "model_unsufficent" : [],
                "check_conditions" : [],
                "rejected" : [],
            },
            "stagnation_flag" : False,
            "optimizer_parameter" : self.optimizer_parameter.copy(),
            "FOM_num_calls": {},
            "dim_Q_r" : [],
            "dim_V_r" : [],
            "counts" : [],
            "inner_loop_statistics" : [],
            "reduced_bases" : None,
            "extention_stats" : {
                "snapshot_projection_error" : {
                    "parameter_basis" : [],
                    "state_basis" : []
                },
            }
        }

    def _append_snapshot_set(self, 
                             snapshots: VectorArray,
                             basis: str,
                             enrichment: Dict) -> None:
 
        preprocess_snapshots_start_time = timer()

        for snapshot in snapshots:
            assert isinstance(snapshot, VectorArray) 
            assert basis in ['parameter_basis','state_basis']

            if basis == 'parameter_basis':
                if not enrichment[basis]['transformation']:
                    self.parameter_shapshots.append(snapshot)
                    return
                
                if enrichment[basis]['transformation']['sample_every_n_th']:
                    n = enrichment[basis]['transformation']['sample_every_n_th']
                    _snapshot = copy.deepcopy(snapshot)
                    assert len(_snapshot) % n == 0
                    assert n == 1 or self.FOM.setup['model_parameter']['q_time_dep']

                    _snapshot = _snapshot[::n]
                    self.parameter_shapshots.append(_snapshot)
                    
                if enrichment[basis]['transformation']['normalize']:
                    norms = self.parameter_shapshots.norm(self.FOM.products['prod_Q'])
                    norms[norms <= 1e-16] = 1
                    self.parameter_shapshots.scal(1/norms)
                

            if basis == 'state_basis':
                if not enrichment[basis]['transformation']:
                    self.state_shapshots.append(_snapshot)
                    return

                if enrichment[basis]['transformation']['sample_every_n_th']:
                    n = enrichment[basis]['transformation']['sample_every_n_th']
                    _snapshot = copy.deepcopy(snapshot)
                    assert ((len(_snapshot))-1) % n == 0

                    _snapshot = _snapshot[::n]
                    self.state_shapshots.append(_snapshot)
                
                if enrichment[basis]['transformation']['normalize']:
                    norms = self.state_shapshots.norm(self.FOM.products['prod_V'])
                    norms[norms <= 1e-16] = 1
                    self.state_shapshots.scal(1/norms)          

        preprocess_snapshots_runtime = timer() - preprocess_snapshots_start_time

        if basis == 'parameter_basis':
            self.statistics["outer_loop_runtime"]['preprocess_parameter_snapshot_runtime'][-1] += preprocess_snapshots_runtime
        elif basis == 'state_basis':
            self.statistics["outer_loop_runtime"]['preprocess_state_snapshot_runtime'][-1] += preprocess_snapshots_runtime
        else:
            raise AttributeError
         
    def _extend_basis_projected_error_HaPOD(self,
                                            snapshots: VectorArray,
                                            basis: str,
                                            product: Operator,
                                            HaPOD_tol: float = 1e-16) -> None:
        assert isinstance(snapshots, VectorArray) 
        assert basis in ['parameter_basis','state_basis']
        assert HaPOD_tol > 0
        assert product.source == product.range == snapshots.space

        if len(self.reductor.bases[basis]) != 0:
            projected_snapshots = self.reductor.bases[basis].lincomb(
                self.reductor.project_vectorarray(snapshots, basis=basis)
            )
            snapshots.axpy(-1,projected_snapshots)
        
        self._extend_basis_snapshot_HaPOD(
            snapshots = snapshots,
            basis = basis,
            product = product,
            HaPOD_tol = HaPOD_tol
        )

    def _extend_basis_snapshot_HaPOD(self,
                                     snapshots: VectorArray,
                                     basis: str,
                                     product: Operator,
                                     HaPOD_tol: float = 1e-16) -> None:
        assert isinstance(snapshots, VectorArray) 
        assert basis in ['parameter_basis','state_basis']
        assert HaPOD_tol > 0
        assert product.source == product.range == snapshots.space

        snapshots, svals, _ = \
        inc_vectorarray_hapod(steps=len(snapshots)/2, 
                              U=snapshots, 
                              eps=HaPOD_tol,
                              omega=0.1,                
                              product=product)
        
        if basis == 'state_basis':
            print("AA")
            print(svals)

        try:
            self.reductor.extend_basis(
                U = snapshots,
                basis = basis
            )
        except ExtensionError:
            self._logger.warning(f"No new vectors were added to {basis}, with tol = {HaPOD_tol}.")

    def _extend_basis_full_HaPOD(self,
                                 snapshots: VectorArray,
                                 basis: str,
                                 product: Operator,
                                 HaPOD_tol: float = 1e-16) -> None:
        
        assert isinstance(snapshots, VectorArray) 
        assert basis in ['parameter_basis','state_basis']
        assert HaPOD_tol > 0
        assert product.source == product.range == snapshots.space

        # print(len(snapshots))
        # self.reductor.bases['state_basis'].append(snapshots)
        # print(len(self.reductor.bases['state_basis']))

        snapshots, svals, _ = \
        inc_vectorarray_hapod(steps=len(self.reductor.bases['state_basis'])/2, 
                              U=self.reductor.bases['state_basis'], 
                              eps=HaPOD_tol,
                              omega=0.1,                
                              product=product)

        print(len(self.reductor.bases['state_basis']))

        if basis == 'state_basis':
            print("AA")
            print(svals)

        self.reductor.bases[basis] = snapshots
    
    def _extend_basis_HaPOD_on_basis(self,
                                     snapshots: VectorArray,
                                     basis: str,
                                     product: Operator,
                                     HaPOD_tol: float = 1e-16) -> None:
        
        assert isinstance(snapshots, VectorArray) 
        assert basis in ['parameter_basis','state_basis']
        assert HaPOD_tol > 0
        assert product.source == product.range == snapshots.space

        snapshots.append(self.reductor.bases[basis])
        snapshots, svals, _ = \
        inc_vectorarray_hapod(steps=len(snapshots)/2, 
                              U=snapshots, 
                              eps=HaPOD_tol,
                              omega=0.1,                
                              product=product)
        
        if basis == 'state_basis':
            print(svals)

        self.reductor.bases[basis] = snapshots
    
    def _extend_basis_last_n_vectors(self,
                                     snapshots: VectorArray,
                                     basis: str,
                                     product: Operator,
                                     HaPOD_tol: float = 1e-16) -> None:
        
        assert isinstance(snapshots, VectorArray) 
        assert basis in ['parameter_basis','state_basis']
        assert HaPOD_tol > 0
        assert product.source == product.range == snapshots.space

        #self.reductor.bases[basis].append(snapshots)
        snapshots, _, _ = \
        inc_vectorarray_hapod(steps=len(snapshots)/2, 
                              U=snapshots, 
                              eps=HaPOD_tol,
                              omega=0.1,                
                              product=product)

        n = 400
        idx = max([n - len(snapshots), 0])
        x = self.FOM.V.empty()
        x.append(self.reductor.bases[basis][-idx:])
        self.reductor.bases[basis] = x

        try:
            self.reductor.extend_basis(
                U = snapshots,
                basis = basis
            )
        except ExtensionError:
            self._logger.warning(f"No new vectors were added to {basis}, with tol = {HaPOD_tol}.")
    
    def extend_bases_and_rebuild_QrVrROM(self,
                                         basis: str,
                                         enrichment : Dict) -> InstationaryModelIP:
        
        parameter_strategy = enrichment['parameter_basis']['strategy']
        parameter_HaPOD_tol = enrichment['parameter_basis']['HaPOD_tol']
        state_strategy = enrichment['state_basis']['strategy']
        state_HaPOD_tol = enrichment['state_basis']['HaPOD_tol']

        assert basis in ['parameter_basis', 'state_basis', 'both']
        #assert parameter_strategy in ['snapshot_HaPOD', 'projected_error_HaPOD', 'full_HaPOD']
        assert parameter_HaPOD_tol > 0
        #assert state_strategy in ['snapshot_HaPOD', 'projected_error_HaPOD', 'full_HaPOD']
        assert state_HaPOD_tol > 0

        self.statistics['extention_stats']['snapshot_projection_error']['parameter_basis'].append(
            self.reductor.calc_projection_error(
                x = self.parameter_shapshots.copy(),
                basis = 'parameter_basis',
                normalize = False
            )
        )
        self.statistics['extention_stats']['snapshot_projection_error']['state_basis'].append(
            self.reductor.calc_projection_error(
                x = self.state_shapshots.copy(),
                basis = 'state_basis',
                normalize = False
            )
        )

        extend_parameter_basis_start_time = timer()
        if basis in ['parameter_basis', 'both']:
            self.logger.debug(f"Extending parameter basis, using {parameter_strategy}, with tol = {parameter_HaPOD_tol}.")
            if parameter_strategy == 'snapshot_HaPOD':
                self._extend_basis_snapshot_HaPOD(
                    snapshots = self.parameter_shapshots,
                    basis='parameter_basis',
                    product=self.FOM.products['prod_Q'],
                    HaPOD_tol = parameter_HaPOD_tol
                )
            elif parameter_strategy == 'projected_error_HaPOD':
                self._extend_basis_projected_error_HaPOD(
                    snapshots = self.parameter_shapshots,
                    basis='parameter_basis',
                    product=self.FOM.products['prod_Q'],
                    HaPOD_tol = parameter_HaPOD_tol
                )
            elif parameter_strategy == 'full_HaPOD':
                self._extend_basis_full_HaPOD(
                    snapshots = self.parameter_shapshots,
                    basis='parameter_basis',
                    product=self.FOM.products['prod_Q'],
                    HaPOD_tol = parameter_HaPOD_tol
                )
                self.reductor.delete_cached_operators()
            elif parameter_strategy == 'HaPOD_on_basis':
                self._extend_basis_HaPOD_on_basis(
                    snapshots = self.parameter_shapshots,
                    basis='parameter_basis',
                    product=self.FOM.products['prod_Q'],
                    HaPOD_tol = parameter_HaPOD_tol
                )
                self.reductor.delete_cached_operators(targets=['A_r'])
            else:
                raise ValueError
            
        self.statistics["outer_loop_runtime"]['extend_parameter_basis_runtime'][-1] += (timer() - extend_parameter_basis_start_time)
        extend_state_basis_start_time = timer()

        if basis in ['state_basis', 'both']:
            self.logger.debug(f"Extending state basis, using {state_strategy}, with tol = {state_HaPOD_tol}.")
            if state_strategy == 'snapshot_HaPOD':
                self._extend_basis_snapshot_HaPOD(
                    snapshots = self.state_shapshots,
                    basis='state_basis',
                    product=self.FOM.products['prod_V'],
                    HaPOD_tol = state_HaPOD_tol
                )
            elif state_strategy == 'projected_error_HaPOD':
                self._extend_basis_projected_error_HaPOD(
                    snapshots = self.state_shapshots,
                    basis='state_basis',
                    product=self.FOM.products['prod_V'],
                    HaPOD_tol = state_HaPOD_tol
                )
            elif state_strategy == 'full_HaPOD':
                self._extend_basis_full_HaPOD(
                    snapshots = self.state_shapshots,
                    basis='state_basis',
                    product=self.FOM.products['prod_V'],
                    HaPOD_tol = state_HaPOD_tol
                )
                self.reductor.delete_cached_operators(targets=['A_r'])

            elif state_strategy == 'last_n_vectors':
                self._extend_basis_last_n_vectors(
                    snapshots = self.state_shapshots,
                    basis='state_basis',
                    product=self.FOM.products['prod_V'],
                    HaPOD_tol = state_HaPOD_tol
                )
                self.reductor.delete_cached_operators(targets=['A_r'])

            elif state_strategy == 'HaPOD_on_basis':
                self._extend_basis_HaPOD_on_basis(
                    snapshots = self.state_shapshots,
                    basis='state_basis',
                    product=self.FOM.products['prod_V'],
                    HaPOD_tol = state_HaPOD_tol
                )
                self.reductor.delete_cached_operators(targets=['A_r'])

            else:
                raise ValueError
        
        print("##########################################")
        snapshots = self.reductor.bases['state_basis']
        snapshots, svals, _ = \
        inc_vectorarray_hapod(steps=len(snapshots)/2, 
                              U=snapshots, 
                              eps=1e-16,
                              omega=0.1,                
                              product=self.FOM.products['prod_V'])
        print(svals)
        
        self.statistics["outer_loop_runtime"]['extend_state_basis_runtime'][-1] += (timer() - extend_state_basis_start_time)
            
        self.reductor.dims_history['parameter_basis'].append(self.reductor.get_bases_dim('parameter_basis'))
        self.reductor.dims_history['state_basis'].append(self.reductor.get_bases_dim('state_basis'))
        
        self.logger.debug(f"Dim Qr-space = {self.reductor.get_bases_dim('parameter_basis')}")
        self.logger.debug(f"Dim Vr-space = {self.reductor.get_bases_dim('state_basis')}")


        reduce_start_time = timer()
        QrVrROM = self.reductor.reduce() 
        self.statistics["outer_loop_runtime"]['reduce_runtime'][-1] += (timer() - reduce_start_time)
        
        return QrVrROM    
       
    def solve(self) -> VectorArray:
        q_0 = self.optimizer_parameter["q_0"].copy()
        alpha_0 = self.optimizer_parameter["alpha_0"]
        tol = self.optimizer_parameter["tol"]
        tau = self.optimizer_parameter["tau"]
        noise_level = self.optimizer_parameter["noise_level"]
        theta = self.optimizer_parameter["theta"]
        Theta = self.optimizer_parameter["Theta"]
        tau_tilde = self.optimizer_parameter["tau_tilde"]

        i_max = self.optimizer_parameter["i_max"]
        reg_loop_max = self.optimizer_parameter["reg_loop_max"]
        i_max_inner = self.optimizer_parameter["i_max_inner"]
        agc_armijo_max_iter = self.optimizer_parameter["agc_armijo_max_iter"]
        TR_armijo_max_iter = self.optimizer_parameter["TR_armijo_max_iter"]


        lin_solver_parms = self.optimizer_parameter['lin_solver_parms']
        use_cached_operators = self.optimizer_parameter['use_cached_operators']        
        enrichment = self.optimizer_parameter['enrichment']

        dump_every_nth_loop = self.optimizer_parameter['dump_every_nth_loop']

        eta0 = self.optimizer_parameter["eta0"]
        kappa_arm = self.optimizer_parameter["kappa_arm"]
        beta_1 = self.optimizer_parameter["beta_1"]
        beta_2 = self.optimizer_parameter["beta_2"]
        beta_3 = self.optimizer_parameter["beta_3"]


        start_time = timer()
        i = 0
        alpha = alpha_0
        delta = noise_level

        solve_snapshot_FOM_start_time = timer()
        q = self.FOM.Q.make_array(q_0)
        u = self.FOM.solve_state(q, use_cached_operators=False)        
        p = self.FOM.solve_adjoint(q, u, use_cached_operators=False)
        J = self.FOM.objective(u)
        nabla_J = self.FOM.gradient(u, p, q, use_cached_operators=False)
        norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)
        self.statistics['outer_loop_runtime']['solve_snapshot_FOM_runtime'].append(timer()  - solve_snapshot_FOM_start_time)
        
        self.statistics["outer_loop_runtime"]['preprocess_parameter_snapshot_runtime'].append(0.0)
        self.statistics["outer_loop_runtime"]['preprocess_state_snapshot_runtime'].append(0.0)
        self.statistics["outer_loop_runtime"]['extend_parameter_basis_runtime'].append(0.0)
        self.statistics["outer_loop_runtime"]['extend_state_basis_runtime'].append(0.0)
        self.statistics["outer_loop_runtime"]['reduce_runtime'].append(0.0)

        #assert norm_nabla_J > 0

        inital_agc_armijo_step_size = 0.5 / norm_nabla_J
        #inital_agc_armijo_step_size = np.min([inital_agc_armijo_step_size, 1])
        inital_agc_armijo_step_size = np.min([inital_agc_armijo_step_size, 1e-2])
        eta = eta0
                    
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
        self.logger.debug(f"  agc_armijo_max_iter : {agc_armijo_max_iter:3.4e}")
        self.logger.debug(f"  TR_armijo_max_iter : {TR_armijo_max_iter:3.4e}")
        self.logger.debug(f"                ")
        self.logger.debug(f"  lin_solver_parms : ")
        for (key,val) in lin_solver_parms.items():
            self.logger.debug(f"        {key} : {val}")
        self.logger.debug(f"  enrichment : ")
        for (key,val) in enrichment.items():
            self.logger.debug(f"        {key} : {val}")
        self.logger.debug(f"  use_cached_operators : {use_cached_operators}")
        self.logger.debug(f"                ")
        self.logger.debug(f"  eta0 : {eta0:3.4e}")
        self.logger.debug(f"  kappa_arm : {kappa_arm:3.4e}")
        self.logger.debug(f"  beta_1 : {beta_1:3.4e}")
        self.logger.debug(f"  beta_2 : {beta_2:3.4e}")
        self.logger.debug(f"  beta_3 : {beta_3:3.4e}")


        self.logger.debug(f"Extending Qr-snapshots")
        self.parameter_shapshots = self.FOM.Q.empty()

        _parameter_shapshots = self.FOM.Q.empty()
        _parameter_shapshots.append(nabla_J)
        _parameter_shapshots.append(q)
        _parameter_shapshots.append(self.FOM.Q.make_array(self.FOM.setup['q_circ']))

        if enrichment['parameter_basis']['include_GN_hessian']:
            GN_hessian = self.FOM.Q.make_array(np.outer(nabla_J.to_numpy()[0], nabla_J.to_numpy()[0]))
            _parameter_shapshots.append(GN_hessian)

            print(GN_hessian)
            print(len(_parameter_shapshots))
        

        B_u = [self.FOM.B(u[idx], idx) for idx in range(len(u))]
        for idx in range(0, self.FOM.nt + 1):
            _parameter_shapshots.append(self.FOM.Q.make_array(B_u[idx].B_u_ad(p[idx])))

        # import matplotlib.pyplot as plt
        # # First image
        # plt.figure()
        # plt.imshow(nabla_J.to_numpy()[0].reshape(31, 31), cmap='viridis')
        # plt.colorbar(label='Value')  # add colorbar on the right
        # plt.title('nabla_J[0]')
        # plt.tight_layout()
        # plt.savefig(self.save_path / 'nabla_J_0.png')
        # plt.close()

        # # Second image
        # for i in [0,500,-1]:
        #     plt.figure()
        #     plt.imshow(GN_hessian.to_numpy()[i].reshape(31,31), cmap='viridis')
        #     plt.colorbar(label='Value')  # add colorbar on the right
        #     plt.title('GN_hessian[0]')
        #     plt.tight_layout()
        #     plt.savefig(self.save_path / f'GN_hessian_{i}.png')
        #     plt.close()

        # import sys
        # sys.exit()


        #_parameter_shapshots.append(self.FOM.Q.make_array(self.FOM.setup['q_exact']))


        self._append_snapshot_set(
            _parameter_shapshots,
            basis='parameter_basis',
            enrichment=enrichment
        )
                  
        self.logger.debug(f"Extending Vr-snapshots")
        self.state_shapshots = self.FOM.V.empty()
        self._append_snapshot_set(
            [u,p],
            basis='state_basis',
            enrichment=enrichment
        )

        self.QrVrROM = self.extend_bases_and_rebuild_QrVrROM(
            basis='both',
            enrichment=enrichment
        )

        q_r = self.reductor.project_vectorarray(q, 'parameter_basis')
        q_r = self.QrVrROM.Q.make_array(q_r)

        u_r = self.QrVrROM.solve_state(q_r)
        p_r = self.QrVrROM.solve_adjoint(q_r, u_r)
        J_r = self.QrVrROM.objective(u_r)
        nabla_J_r = self.QrVrROM.gradient(u_r, p_r, q_r)
        norm_nabla_J_r = self.QrVrROM.compute_gradient_norm(nabla_J_r)

        abs_est_error_J_r, abs_est_error_nabla_J_r = self.estimate_objective_error(
            model=self.QrVrROM,
            q = q_r,
            u = u_r,
            p = p_r,
            use_cached_operators=use_cached_operators
        )

        if J_r > 0:
            rel_est_error_J_r = abs_est_error_J_r / J_r
        else:
            rel_est_error_J_r = np.inf

        if norm_nabla_J_r > 0:
            rel_est_error_nabla_J_r = abs_est_error_nabla_J_r / norm_nabla_J_r
        else:
            rel_est_error_nabla_J_r = np.inf

        self.statistics["q"].append(q)
        self.statistics["alpha"].append(alpha)
        self.statistics["J"].append(J)
        self.statistics["norm_nabla_J"].append(norm_nabla_J)
        self.statistics["J_r"].append(J_r)
        self.statistics['abs_est_error_J_r'].append(abs_est_error_J_r)
        self.statistics['rel_est_error_J_r'].append(rel_est_error_J_r)
        self.statistics['abs_est_error_nabla_J_r'].append(abs_est_error_nabla_J_r)
        self.statistics['rel_est_error_nabla_J_r'].append(rel_est_error_nabla_J_r)
        self.statistics['dim_Q_r'].append(self.reductor.get_bases_dim('parameter_basis'))
        self.statistics['dim_V_r'].append(self.reductor.get_bases_dim('state_basis'))

        convergence_criterium = np.sqrt(2 * J) < tol+tau*noise_level
        AGC_jump_back = False

        while not convergence_criterium and i<i_max:            
            outer_loop_start_time = timer()
            self.logger.info(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            self.logger.warning(f"Qr-Vr-IRGNM iteration {i}: J = {J:3.4e} is not sufficent: {np.sqrt(2 * J):3.4e} > {(tol+tau*noise_level):3.4e}.")
            self.logger.info(f'Start Qr-Vr-IRGNM iteration {i}: J = {J:3.4e}, norm_nabla_J = {norm_nabla_J:3.4e}, alpha = {alpha:1.4e}')
            self.logger.info(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            q_r = self.reductor.project_vectorarray(q, 'parameter_basis')
            q_r = self.QrVrROM.Q.make_array(q_r)

            u_r = self.QrVrROM.solve_state(q_r, use_cached_operators=use_cached_operators)
            p_r = self.QrVrROM.solve_adjoint(q_r, u_r, use_cached_operators=use_cached_operators)
            J_r = self.QrVrROM.objective(u_r)
            
            nabla_J_r = self.QrVrROM.gradient(u_r, p_r, q_r, use_cached_operators=use_cached_operators)
            
            abs_est_error_J_r, abs_est_error_nabla_J_r = self.estimate_objective_error(
                model = self.QrVrROM,
                q = q_r,
                u = u_r,
                p = p_r,
                use_cached_operators=use_cached_operators)
            
            if J_r > 0:
                rel_est_error_J_r = abs_est_error_J_r / J_r
            else:
                rel_est_error_J_r = np.inf

            print(rel_est_error_J_r)
            print(abs_est_error_J_r)
            print(J_r)

            proj_q_in_tr = rel_est_error_J_r <= eta
        
            if AGC_jump_back: 
                self.statistics['flags']['proj_q_in_tr'][-1] = proj_q_in_tr
            else:
                self.statistics['flags']['proj_q_in_tr'].append(proj_q_in_tr)

            if not proj_q_in_tr:
                # assert enrichment['parameter_basis']['strategy'] in ['snapshot_HaPOD', 'projected_error_HaPOD']
                # assert enrichment['state_basis']['strategy'] in ['snapshot_HaPOD', 'projected_error_HaPOD']

                self._logger.warning(f"q^(i) is not in the trust region.")
                self._logger.warning(f"Extending reduced spaces with all snapshots.")

                _enrichment = copy.deepcopy(enrichment)
                _enrichment['parameter_basis']['HaPOD_tol'] = MACHINE_EPS
                _enrichment['parameter_basis']['transformation']['sample_every_n_th'] = 1
                _enrichment['state_basis']['HaPOD_tol'] = MACHINE_EPS
                _enrichment['state_basis']['transformation']['sample_every_n_th'] = 1

                self.logger.debug(f"Extending Qr-snapshots")
                self.parameter_shapshots = self.FOM.Q.empty()        
                self._append_snapshot_set(
                    [nabla_J],
                    basis='parameter_basis',
                    enrichment=_enrichment
                )
                
                self.logger.debug(f"Extending Vr-snapshots")
                self.state_shapshots = self.FOM.V.empty()
                self._append_snapshot_set(
                    [u,p],
                    basis='state_basis',
                    enrichment=_enrichment
                )

                self.QrVrROM = self.extend_bases_and_rebuild_QrVrROM(
                    basis='both',
                    enrichment = _enrichment
                )

                q_r = self.reductor.project_vectorarray(q, 'parameter_basis')
                q_r = self.QrVrROM.Q.make_array(q_r)
                u_r = self.QrVrROM.solve_state(q_r, use_cached_operators=False)
                p_r = self.QrVrROM.solve_adjoint(q_r, u_r, use_cached_operators=False)
                J_r = self.QrVrROM.objective(u_r)
                nabla_J_r = self.QrVrROM.gradient(u_r, p_r, q_r)
                norm_nabla_J_r = self.QrVrROM.compute_gradient_norm(nabla_J_r)

                abs_est_error_J_r, _ = self.estimate_objective_error(
                    model = self.QrVrROM,
                    q = q_r,
                    u = u_r,
                    p = p_r,
                    use_cached_operators=use_cached_operators)
                
                if J_r > 0:
                    rel_est_error_J_r = abs_est_error_J_r / J_r
                else:
                    rel_est_error_J_r = np.inf
            
            assert (rel_est_error_J_r - 1e-16) <= eta 

            IRGNM_statistic = None
            projector = SimpleBoundDomainProjector(
                model = self.QrVrROM,
                bounds = self.FOM.bounds,
                reductor = self.reductor,
                use_sufficient_condition = True,
                logger = self.logger
            )

            ########################################### AGC ###########################################

            self.logger.warning("Calculate AGC with Armijo backtracking.")

            AGC_start_time = timer()

            q_agc, J_r_AGC, model_unsufficent, AGC_max_iter_cond, _ = self._armijo_TR_line_serach(
                model = self.QrVrROM,
                previous_q = q_r,
                previous_J = J_r,
                search_direction = -nabla_J_r,
                max_iter = agc_armijo_max_iter,
                inital_step_size = inital_agc_armijo_step_size,
                eta = eta,
                beta = beta_1,
                kappa_arm = kappa_arm,
                use_cached_operators=use_cached_operators,
                projector=projector
            )

            AGC_decay_cond = J_r_AGC < (J + 1e-13)

            if not AGC_jump_back:
                self.statistics['flags']['AGC_decay_cond'].append(AGC_decay_cond)
                        
            if not AGC_decay_cond:
                self._logger.warning(f"J_r_AGC = {J_r_AGC:3.4e} is greater or equal than J = {J:3.4e}.")
                self._logger.warning(f"Extending reduced spaces with all snapshots and recomputing AGC.")

                
                _enrichment = copy.deepcopy(enrichment)
                _enrichment['parameter_basis']['HaPOD_tol'] = MACHINE_EPS
                _enrichment['parameter_basis']['transformation']['sample_every_n_th'] = 1
                _enrichment['state_basis']['HaPOD_tol'] = MACHINE_EPS
                _enrichment['state_basis']['transformation']['sample_every_n_th'] = 1

                self.logger.debug(f"Extending Qr-snapshots")
                self.parameter_shapshots = self.FOM.Q.empty()
                self._append_snapshot_set(
                    [nabla_J],
                    basis='parameter_basis',
                    enrichment=_enrichment
                )
                            
                self.logger.debug(f"Extending Vr-snapshots")
                self.state_shapshots = self.FOM.V.empty()
                self._append_snapshot_set(
                    [u,p],
                    basis='state_basis',
                    enrichment=_enrichment
                )
            
                self.QrVrROM = self.extend_bases_and_rebuild_QrVrROM(
                    basis='both',
                    enrichment = _enrichment
                )
                AGC_jump_back = True
                continue
             
            assert not AGC_max_iter_cond

            AGC_jump_back = False
                
            self.statistics['flags']['model_unsufficent'].append(model_unsufficent)
            self.statistics["outer_loop_runtime"]['AGC_runtime'].append(timer() - AGC_start_time)

            q_r = q_agc.copy()
            ########################################### IRGNM ###########################################
            IRGNM_start_time = timer()

            TR_backtracking_params = {
                'max_iter' : TR_armijo_max_iter, 
                'inital_step_size' : 1, 
                'eta' : eta, 
                'beta' : beta_1, 
                "kappa_arm" : kappa_arm
            }
            #projector = None

            if not model_unsufficent:
                q_r, IRGNM_statistic = self.IRGNM(model = self.QrVrROM,
                                                  q_0 = q_r,
                                                  alpha_0 = alpha,
                                                  tol = tol,
                                                  tau = tau,
                                                  noise_level = delta,
                                                  i_max = i_max_inner,
                                                  theta = theta,
                                                  Theta = Theta,
                                                  reg_loop_max = reg_loop_max,
                                                  use_TR=True,
                                                  TR_backtracking_params=TR_backtracking_params,
                                                  lin_solver_parms=lin_solver_parms,
                                                  use_cached_operators=use_cached_operators,
                                                  projector=projector)

            self.statistics["outer_loop_runtime"]['IRGNM_runtime'].append(timer() - IRGNM_start_time)

            ########################################### Accept / Reject ###########################################

            if IRGNM_statistic is not None:
                check_conditions = len(IRGNM_statistic['q']) > 1
            else:
                check_conditions = False

            self.statistics['flags']['check_conditions'].append(check_conditions)

            if check_conditions:
                self.logger.debug("Decide on q; Either accept or reject")

                u_r = self.QrVrROM.solve_state(q_r)
                p_r = self.QrVrROM.solve_adjoint(q_r, u_r)
                J_r = self.QrVrROM.objective(u_r)
                nabla_J_r = self.QrVrROM.gradient(u_r, p_r, q_r)
                norm_nabla_J_r = self.QrVrROM.compute_gradient_norm(nabla_J_r)

                abs_est_error_J_r, abs_est_error_nabla_J_r = self.estimate_objective_error(
                    model = self.QrVrROM,
                    q=q_r,
                    u = u_r,
                    p = p_r,
                    use_cached_operators=use_cached_operators
                )
                
                if J_r > 0:
                    rel_est_error_J_r = abs_est_error_J_r / J_r
                else:
                    rel_est_error_J_r = np.inf

                if norm_nabla_J_r > 0:
                    rel_est_error_nabla_J_r = abs_est_error_nabla_J_r / norm_nabla_J_r
                else:
                    rel_est_error_nabla_J_r = np.inf
                
                if abs_est_error_J_r <= MACHINE_EPS:
                    abs_est_error_J_r = 0.0

                sufficent_condition = J_r + abs_est_error_J_r < J_r_AGC        
                necessary_condition = J_r - abs_est_error_J_r <= J_r_AGC

                self.logger.debug(f"    J_r_AGC = {J_r_AGC:3.4e}")
                self.logger.debug(f"    J_r = {J_r:3.4e}")
                self.logger.debug(f"    abs_est_error_J_r = {abs_est_error_J_r:3.4e}")
                self.logger.debug(f"    J_r + abs_est_error_J_r = {J_r + abs_est_error_J_r:3.4e}; sufficent_condition = {sufficent_condition}")
                self.logger.debug(f"    J_r - abs_est_error_J_r = {J_r - abs_est_error_J_r:3.4e}; necessary_condition = {necessary_condition}")

                rejected = False
                
                if sufficent_condition:
                    self.logger.info(f"    Accept q.")
                    rejected = False

                    solve_snapshot_FOM_start_time = timer()
                    q = self.reductor.reconstruct(q_r, basis='parameter_basis')
                    u = self.FOM.solve_state(q)
                    p = self.FOM.solve_adjoint(q, u)
                    J = self.FOM.objective(u)
                    nabla_J = self.FOM.gradient(u, p, q)
                    norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)
                    self.statistics['outer_loop_runtime']['solve_snapshot_FOM_runtime'].append(timer()  - solve_snapshot_FOM_start_time)

                    delta_J = self.statistics["J"][-1] - J
                    delta_J_r = self.statistics["J_r"][-1]-J_r

                    if delta_J_r > 0:
                        rho = delta_J / delta_J_r
                    else:
                        rho = np.inf

                    print("rho = ")
                    print(rho)

                    if rho > beta_2:
                        eta = 1/ beta_3 * eta

                elif not necessary_condition:
                    self.logger.info(f"    Reject q.")
                    rejected = True
                    # q remain unchanged
                    eta = beta_3 * eta
                    
                    solve_snapshot_FOM_start_time = timer()
                    self.statistics['outer_loop_runtime']['solve_snapshot_FOM_runtime'].append(timer()  - solve_snapshot_FOM_start_time)
                else:
                    solve_snapshot_FOM_start_time = timer()
                    q_ = self.reductor.reconstruct(q_r, basis='parameter_basis')
                    u_ = self.FOM.solve_state(q_, use_cached_operators=use_cached_operators)
                    p_ = self.FOM.solve_adjoint(q_, u_, use_cached_operators=use_cached_operators)
                    J_ = self.FOM.objective(u_)
                    nabla_J_ = self.FOM.gradient(u_, p_, q, use_cached_operators=use_cached_operators)
                    norm_nabla_J_ = self.FOM.compute_gradient_norm(nabla_J_)
                    self.statistics['outer_loop_runtime']['solve_snapshot_FOM_runtime'].append(timer()  - solve_snapshot_FOM_start_time)
                    
                    EASDC = J_ <= J_r_AGC
                    self.logger.info(f"    J = {J:3.4e}; EASDC = {EASDC}.")
                    
                    if EASDC:
                        self.logger.info(f"    Accept q.")
                        rejected = False

                        q = q_
                        u = u_
                        p = p_
                        J = J_
                        nabla_J = nabla_J_
                        norm_nabla_J = norm_nabla_J_
                        
                        delta_J = self.statistics["J"][-1] - J
                        delta_J_r = self.statistics["J_r"][-1] - J_r

                        if delta_J_r > 0:
                            rho = delta_J / delta_J_r
                        else:
                            rho = np.inf

                        if rho > beta_2:
                            eta = 1/ beta_3 * eta
                    else:
                        self.logger.info(f"    Reject q.")
                        # q remain unchanged
                        rejected = True
                        eta = beta_3 * eta
                    
                    self.logger.info(f"    eta = {eta:3.4e}.")
            else:
                self.logger.debug("Not found q_trial; Using AGC.")
                rejected = False
                q_r = q_agc.copy()

                solve_snapshot_FOM_start_time = timer()
                q = self.reductor.reconstruct(q_r, basis='parameter_basis')
                u = self.FOM.solve_state(q, use_cached_operators=use_cached_operators)
                p = self.FOM.solve_adjoint(q, u, use_cached_operators=use_cached_operators)
                J = self.FOM.objective(u)
                self.statistics['outer_loop_runtime']['solve_snapshot_FOM_runtime'].append(timer()  - solve_snapshot_FOM_start_time)

                nabla_J = self.FOM.gradient(u, p, q, use_cached_operators=use_cached_operators)
                norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)
                eta = beta_3 * eta

            self.statistics["outer_loop_runtime"]['preprocess_parameter_snapshot_runtime'].append(0.0)
            self.statistics["outer_loop_runtime"]['preprocess_state_snapshot_runtime'].append(0.0)
            self.statistics["outer_loop_runtime"]['extend_parameter_basis_runtime'].append(0.0)
            self.statistics["outer_loop_runtime"]['extend_state_basis_runtime'].append(0.0)
            self.statistics["outer_loop_runtime"]['reduce_runtime'].append(0.0)

            ########################################### Final ###########################################

            convergence_criterium = np.sqrt(2 * J) < tol+tau*noise_level
            self.statistics['flags']['rejected'].append(rejected)

            if not rejected:
                delta = delta
                
                if IRGNM_statistic is not None:
                    try:
                        alpha = IRGNM_statistic["alpha"][1]
                        #alpha = IRGNM_statistic["alpha"][-1]
                    except IndexError:
                        pass
                    
                if not convergence_criterium:
                    self.logger.debug(f"Extending Qr-snapshots")
                    self.parameter_shapshots = self.FOM.Q.empty()

                    _parameter_shapshots = self.FOM.Q.empty()
                    _parameter_shapshots.append(nabla_J)
                    
                    if enrichment['parameter_basis']['include_GN_hessian']:
                        GN_hessian = self.FOM.Q.make_array(np.outer(nabla_J.to_numpy()[0], nabla_J.to_numpy()[0]))
                        _parameter_shapshots.append(GN_hessian)

                    B_u = [self.FOM.B(u[idx], idx) for idx in range(len(u))]
                    for idx in range(0, self.FOM.nt + 1):
                        _parameter_shapshots.append(self.FOM.Q.make_array(B_u[idx].B_u_ad(p[idx])))


                    #_parameter_shapshots.append(self.FOM.Q.make_array(self.FOM.setup['q_exact']))

                    self._append_snapshot_set(
                        _parameter_shapshots,
                        basis='parameter_basis',
                        enrichment=enrichment
                    )
                                
                    self.logger.debug(f"Extending Vr-snapshots")
                    self.state_shapshots = self.FOM.V.empty()
                    self._append_snapshot_set(
                        [u,p],
                        basis='state_basis',
                        enrichment=enrichment
                    )

                    self.QrVrROM = self.extend_bases_and_rebuild_QrVrROM(
                        basis='both',
                        enrichment=enrichment
                    )

                    q_r = self.reductor.project_vectorarray(q, 'parameter_basis')
                    q_r = self.QrVrROM.Q.make_array(q_r)

                self.statistics["q"].append(q)
                self.statistics["alpha"].append(alpha)
                self.statistics["J"].append(J)
                self.statistics["norm_nabla_J"].append(norm_nabla_J)
                self.statistics["J_r"].append(J_r)
                self.statistics['abs_est_error_J_r'].append(abs_est_error_J_r)
                self.statistics['rel_est_error_J_r'].append(rel_est_error_J_r)
                self.statistics['abs_est_error_nabla_J_r'].append(abs_est_error_nabla_J_r)
                self.statistics['rel_est_error_nabla_J_r'].append(rel_est_error_nabla_J_r)
                self.statistics['dim_Q_r'].append(self.reductor.get_bases_dim('parameter_basis'))
                self.statistics['dim_V_r'].append(self.reductor.get_bases_dim('state_basis'))
                self.statistics["counts"].append(IRGNM_statistic['counts'])
                self.statistics["inner_loop_statistics"].append(IRGNM_statistic)
                self.statistics["total_runtime"].append(timer() - start_time)    
                self.statistics["outer_loop_runtime"]['total_runtime'].append(timer() - outer_loop_start_time)

            if (i % dump_every_nth_loop == 0) or (i == 1):
                self.dump_stats(data=self.statistics,
                                save_path = self.save_path / f'TR_IRGNM_{i}.pkl')
        
            if i > 3:
                buffer = self.statistics["J"][-3:]
                if abs(buffer[0] - buffer[1]) / abs(buffer[0]) < STAGNATION_TOL and abs(buffer[1] - buffer[2]) / abs(buffer[1])< STAGNATION_TOL:
                    self.statistics["stagnation_flag"] = True
                    self.logger.info(f"Stop at iteration {i+1} of {int(i_max)}, due to stagnation.")
                    break

            if convergence_criterium:
                break

            i += 1    

        #self.statistics["total_runtime"].append(timer() - start_time)
        self.statistics["FOM_num_calls"] = self.FOM.num_calls
        self.statistics["reduced_bases"] = self.reductor.bases

        data = self.statistics
        data = self.dump_prepare_statistics(data)
        self.dump_stats(data=data,
                        save_path = self.save_path / f'TR_IRGNM_final.pkl')
        return q

