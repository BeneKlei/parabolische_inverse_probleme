import logging
import numpy as np
from abc import abstractmethod
from typing import Dict, Union, Tuple
from timeit import default_timer as timer
from pathlib import Path
import scipy.sparse.linalg as spla

from pymor.vectorarrays.interface import VectorArray
from pymor.algorithms.hapod import inc_vectorarray_hapod
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.operators.interface import Operator
from pymor.core.base import BasicObject

from RBInvParam.model import InstationaryModelIP
from RBInvParam.linear_solver.gradient_descent import gradient_descent_linearized_problem
from RBInvParam.linear_solver.BiCGSTAB import BiCGStab_linearized_problem
from RBInvParam.reductor import InstationaryModelIPReductor
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.utils.io import save_dict_to_pkl
from RBInvParam.domain_projector import SimpleBoundDomainProjector


MACHINE_EPS = 1e-16
STAGNATION_TOL = 1e-3

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
        search_direction.scal(1.0 / model.compute_gradient_norm(search_direction))

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
            J_rel_error = model.estimate_objective_error(
                q=current_q,
                u = u,
                p = p,
                use_cached_operators=use_cached_operators) / current_J
        else:
            J_rel_error = np.inf
        
        
        TR_condition = J_rel_error <= eta
        condition = armijo_condition & TR_condition
        i += 1

        print(lhs)
        print(rhs)
        print(step_size)

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
            
            if abs(lhs) <= MACHINE_EPS:
                lhs = 0

            if abs(rhs) <= MACHINE_EPS:
                rhs = 0

            print("############")
            print(lhs)
            print(rhs)
            print(step_size)

            armijo_condition = lhs >= rhs

            if current_J > 0:                
                J_rel_error = model.estimate_objective_error(
                    q=current_q,
                    u = u,
                    p = p,
                    use_cached_operators=use_cached_operators) / current_J
            else:
                J_rel_error = np.inf

            TR_condition = J_rel_error <= eta
            condition = armijo_condition & TR_condition
            
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
        
        if not q_0 in self.FOM.Q:
            q_ = self.reductor.reconstruct(q_0, basis='parameter_basis')
        else:
            q_ = q_0.copy()

        q_ = q_.to_numpy().flatten()
        mask_lb = q_ >= self.FOM.bounds[:,0]
        mask_ub = q_ <= self.FOM.bounds[:,1]
        assert np.all(mask_lb) and np.all(mask_ub)

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

                TR_backtracking_params['inital_step_size'] = np.min([step_size * 2, 1])

                if model_unsufficent or TR_max_iter_cond:
                    break
                else:
                    q = q_TR
            else:
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
    
    def _HaPOD(self, 
               shapshots: VectorArray, 
               basis: str,
               product: Operator,
               eps: float = 1e-17) -> Tuple[VectorArray, np.array]:
            
        assert basis in ['parameter_basis', 'state_basis']

        if basis == 'parameter_basis':
            norms = shapshots.norm(self.FOM.products['prod_Q'])
        elif basis == 'state_basis':
            norms = shapshots.norm(self.FOM.products['prod_V'])
        else:
            raise ValueError
        
        shapshots.scal(1/norms)

        if len(self.reductor.bases[basis]) != 0:
            projected_shapshots = self.reductor.bases[basis].lincomb(
                self.reductor.project_vectorarray(shapshots, basis=basis)
            )
            shapshots.axpy(-1,projected_shapshots)
                        
        shapshots, svals, _ = \
        inc_vectorarray_hapod(steps=len(shapshots)/5, 
                              U=shapshots, 
                              eps=eps,
                              omega=0.1,                
                              product=product)
        
        if basis == 'parameter_basis':
            norms = shapshots.norm(self.FOM.products['prod_Q'])
        elif basis == 'state_basis':
            norms = shapshots.norm(self.FOM.products['prod_V'])
        else:
            raise ValueError
        
        #shapshots.scal(1/norms)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(svals)

        # # For orthogonalization
        # # TODO Check
        # shapshots, svals, _ = \
        # inc_vectorarray_hapod(steps=len(shapshots)/5, 
        #                       U=shapshots, 
        #                       eps=eps,
        #                       omega=0.1,
        #                       product=product)


        return shapshots, svals

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
                                        dump_every_nth_loop=dump_every_nth_loop)

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

        while np.sqrt(2 * J) >= tol+tau*noise_level and i<i_max:
            self.logger.info(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            self.logger.warning(f"Qr-IRGNM iteration {i}: J = {J:3.4e} is not sufficent: {np.sqrt(2 * J):3.4e} > {(tol+tau*noise_level):3.4e}.")
            self.logger.info(f'Start Qr-IRGNM iteration {i}: J = {J:3.4e}, norm_nabla_J = {norm_nabla_J:3.4e}, alpha = {alpha:1.4e}')
            self.logger.info(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            q_r = self.reductor.project_vectorarray(q, 'parameter_basis')
            q_r = self.QrFOM.Q.make_array(q_r)
            
            self.IRGNM_idx += 1
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
        )
        self.QrVrROM = None

        self.statistics = {
            "q" : [],
            "alpha" : [],
            "J" : [],
            "norm_nabla_J" : [],
            "J_r" : [],
            "abs_est_error_J_r" : [],
            "rel_est_error_J_r" : [],
            "total_runtime" : [],
            "stagnation_flag" : False,
            "optimizer_parameter" : self.optimizer_parameter.copy(),
            "FOM_num_calls": {},
            "dim_Q_r" : [],
            "dim_V_r" : [],
            "counts" : [],
            "reduced_bases" : None
        }

    def solve(self) -> VectorArray :
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
        HaPOD_tol = self.optimizer_parameter['HaPOD_tol']

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

        q = self.FOM.Q.make_array(q_0)
        u = self.FOM.solve_state(q, use_cached_operators=use_cached_operators)        
        p = self.FOM.solve_adjoint(q, u, use_cached_operators=use_cached_operators)
        J = self.FOM.objective(u)
        nabla_J = self.FOM.gradient(u, p, q, use_cached_operators=use_cached_operators)

        norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)
        assert norm_nabla_J > 0

        inital_agc_armijo_step_size = 0.5 / norm_nabla_J
        inital_agc_armijo_step_size = np.min([inital_agc_armijo_step_size, 1])
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
        self.logger.debug(f"  use_cached_operators : {use_cached_operators}")
        self.logger.debug(f"  HaPOD_tol : {HaPOD_tol}")
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
                                                 product=self.FOM.products['prod_Q'],
                                                 eps=HaPOD_tol)

        self.reductor.extend_basis(
             U = parameter_shapshots,
             basis = 'parameter_basis'
        )

        self.logger.debug(f"Extending Vr-space")
        state_shapshots = self.FOM.V.empty()
        state_shapshots.append(u)
        state_shapshots.append(p)

        self.logger.debug(f"Performing HaPOD on state snapshots.")
        state_shapshots, _ = self._HaPOD(shapshots=state_shapshots, 
                                         basis='state_basis',
                                         product=self.FOM.products['prod_V'],
                                         eps=HaPOD_tol)
        
        self.reductor.extend_basis(
             U = state_shapshots,
             basis = 'state_basis'
        )

        self.QrVrROM = self.reductor.reduce()
        
        self.logger.debug(f"Dim Qr-space = {self.reductor.get_bases_dim('parameter_basis')}")
        self.logger.debug(f"Dim Vr-space = {self.reductor.get_bases_dim('state_basis')}")

        q_r = self.reductor.project_vectorarray(q, 'parameter_basis')
        q_r = self.QrVrROM.Q.make_array(q_r)

        u_r = self.QrVrROM.solve_state(q_r)
        p_r = self.QrVrROM.solve_adjoint(q_r, u_r)
        J_r = self.QrVrROM.objective(u_r)

        abs_est_error_J_r = self.QrVrROM.estimate_objective_error(
                q=q_r,
                u = u_r,
                p = p_r,
                use_cached_operators=use_cached_operators)
        
        if J_r > 0:
            rel_est_error_J_r = abs_est_error_J_r / J_r
        else:
            rel_est_error_J_r = np.inf

        self.statistics["q"].append(q)
        self.statistics["alpha"].append(alpha)
        self.statistics["J"].append(J)
        self.statistics["norm_nabla_J"].append(norm_nabla_J)
        self.statistics["J_r"].append(J_r)
        self.statistics['abs_est_error_J_r'].append(abs_est_error_J_r)
        self.statistics['rel_est_error_J_r'].append(rel_est_error_J_r)
        self.statistics['dim_Q_r'].append(self.reductor.get_bases_dim('parameter_basis'))
        self.statistics['dim_V_r'].append(self.reductor.get_bases_dim('state_basis'))

        convergence_criterium = np.sqrt(2 * J) < tol+tau*noise_level

        while not convergence_criterium and i<i_max:
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
            assert not AGC_max_iter_cond
            q_r = q_agc.copy()
            ########################################### IRGNM ###########################################
            TR_backtracking_params = {
                'max_iter' : TR_armijo_max_iter, 
                'inital_step_size' : 1, 
                'eta' : eta, 
                'beta' : beta_1, 
                "kappa_arm" : kappa_arm
            }
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

            ########################################### Accept / Reject ###########################################

            if IRGNM_statistic is not None:
                check_conditions = len(IRGNM_statistic['q']) > 1
            else:
                check_conditions = False

            if check_conditions:
                self.logger.debug("Decide on q; Either accept or reject")

                u_r = self.QrVrROM.solve_state(q_r)
                p_r = self.QrVrROM.solve_adjoint(q_r, u_r)
                J_r = self.QrVrROM.objective(u_r)

                abs_est_error_J_r = self.QrVrROM.estimate_objective_error(
                        q=q_r,
                        u = u_r,
                        p = p_r,
                        use_cached_operators=use_cached_operators)
                
                if J_r > 0:
                    rel_est_error_J_r = abs_est_error_J_r / J_r
                else:
                    rel_est_error_J_r = np.inf
                
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

                    q = self.reductor.reconstruct(q_r, basis='parameter_basis')
                    u = self.FOM.solve_state(q)
                    p = self.FOM.solve_adjoint(q, u)
                    J = self.FOM.objective(u)

                    nabla_J = self.FOM.gradient(u, p, q)
                    norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)

                    delta_J = self.statistics["J"][-1] -J
                    delta_J_r = self.statistics["J_r"][-1]-J_r

                    if delta_J_r > 0:
                        rho = delta_J / delta_J_r
                    else:
                        rho = np.inf

                    if rho > beta_2:
                        eta = 1/ beta_3 * eta

                elif not necessary_condition:
                    self.logger.info(f"    Reject q.")
                    rejected = True
                    # q remain unchanged
                    eta = beta_3 * eta
                else:
                    q = self.reductor.reconstruct(q_r, basis='parameter_basis')
                    u = self.FOM.solve_state(q, use_cached_operators=use_cached_operators)
                    p = self.FOM.solve_adjoint(q, u, use_cached_operators=use_cached_operators)
                    J = self.FOM.objective(u)
                    nabla_J = self.FOM.gradient(u, p, q, use_cached_operators=use_cached_operators)
                    norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)
                    
                    EASDC = J <= J_r_AGC
                    self.logger.info(f"    J = {J:3.4e}; EASDC = {EASDC}.")
                    
                    if EASDC:
                        self.logger.info(f"    Accept q.")
                        rejected = False
                        # q = self.reductor.reconstruct(q_r, basis='parameter_basis')
                        # u = self.FOM.solve_state(q, use_cached_operators=use_cached_operators)
                        # p = self.FOM.solve_adjoint(q, u, use_cached_operators=use_cached_operators)
                        # J = self.FOM.objective(u)
                        # nabla_J = self.FOM.gradient(u, p, q, use_cached_operators=use_cached_operators)
                        # norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)

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
                q = self.reductor.reconstruct(q_r, basis='parameter_basis')
                u = self.FOM.solve_state(q, use_cached_operators=use_cached_operators)
                p = self.FOM.solve_adjoint(q, u, use_cached_operators=use_cached_operators)
                J = self.FOM.objective(u)
                nabla_J = self.FOM.gradient(u, p, q, use_cached_operators=use_cached_operators)
                norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)
                eta = beta_3 * eta

            ########################################### Final ###########################################

            convergence_criterium = np.sqrt(2 * J) < tol+tau*noise_level

            if not rejected:
                delta = delta
                
                if IRGNM_statistic is not None:
                    try:
                        alpha = IRGNM_statistic["alpha"][1]
                    except IndexError:
                        pass
                    
                if not convergence_criterium:
                    self.logger.debug(f"Extending Qr-space")
                    parameter_shapshots = self.FOM.Q.empty()
                    parameter_shapshots.append(nabla_J)
                    
                    parameter_shapshots.append(q)

                    if self.FOM.setup['model_parameter']['q_time_dep']:
                        self.logger.debug(f"Performing HaPOD on parameter snapshots.")
                        parameter_shapshots, _ = self._HaPOD(shapshots=parameter_shapshots, 
                                                            basis='parameter_basis',
                                                            product=self.FOM.products['prod_Q'],
                                                            eps=HaPOD_tol)
                    self.reductor.extend_basis(
                        U = parameter_shapshots,
                        basis = 'parameter_basis'
                    )
                    self.logger.debug(f"Extending Vr-space")
                    state_shapshots = self.FOM.V.empty()
                    state_shapshots.append(u)
                    state_shapshots.append(p)

                    self.logger.debug(f"Performing HaPOD on state snapshots.")
                    state_shapshots, _ = self._HaPOD(shapshots=state_shapshots, 
                                                    basis='state_basis',
                                                    product=self.FOM.products['prod_V'],
                                                    eps=HaPOD_tol)
                    
                    self.reductor.extend_basis(
                        U = state_shapshots,
                        basis = 'state_basis'
                    )
                    self.QrVrROM = self.reductor.reduce()
                    q_r = self.reductor.project_vectorarray(q, 'parameter_basis')
                    q_r = self.QrVrROM.Q.make_array(q_r)
                    self.logger.debug(f"Dim Qr-space = {self.reductor.get_bases_dim('parameter_basis')}")
                    self.logger.debug(f"Dim Vr-space = {self.reductor.get_bases_dim('state_basis')}")

                self.statistics["q"].append(q)
                self.statistics["alpha"].append(alpha)
                self.statistics["J"].append(J)
                self.statistics["norm_nabla_J"].append(norm_nabla_J)
                self.statistics["J_r"].append(J_r)
                self.statistics['abs_est_error_J_r'].append(abs_est_error_J_r)
                self.statistics['rel_est_error_J_r'].append(rel_est_error_J_r)
                self.statistics['dim_Q_r'].append(self.reductor.get_bases_dim('parameter_basis'))
                self.statistics['dim_V_r'].append(self.reductor.get_bases_dim('state_basis'))
                self.statistics["counts"].append(IRGNM_statistic['counts'])
            
            if convergence_criterium:
                break

            if i > 3:
                buffer = self.statistics["J"][-3:]
                if abs(buffer[0] - buffer[1]) / abs(buffer[0]) < STAGNATION_TOL and abs(buffer[1] - buffer[2]) / abs(buffer[1])< STAGNATION_TOL:
                    self.statistics["stagnation_flag"] = True
                    self.logger.info(f"Stop at iteration {i+1} of {int(i_max)}, due to stagnation.")
                    break
            
            self.statistics["FOM_num_calls"] = self.FOM.num_calls
            if (i % dump_every_nth_loop == 0) or (i == 1):
                self.dump_stats(data=self.statistics,
                                save_path = self.save_path / f'TR_IRGNM_{i}.pkl')

            i += 1    
            self.statistics["total_runtime"].append(timer() - start_time)    

        self.statistics["total_runtime"].append(timer() - start_time)
        self.statistics["FOM_num_calls"] = self.FOM.num_calls
        self.statistics["reduced_bases"] = self.reductor.bases

        self.dump_stats(data=self.statistics,
                        save_path = self.save_path / f'TR_IRGNM_final.pkl')
        return q

