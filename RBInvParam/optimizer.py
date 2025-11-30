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
from RBInvParam.snapshot_preprocessor import SnapshotPreprocessor
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.utils.io import save_dict_to_pkl, dealii_vector_space_to_numpy
from RBInvParam.domain_projector import SimpleBoundDomainProjector
 

MACHINE_EPS = 1e-16
STAGNATION_TOL = 1e-6

#######################################################################

error_estimate_targets_outer = ['J']
error_estimate_targets_inner = ['J']

#error_estimate_targets_outer = ['J', 'nabla_J']
#error_estimate_targets_inner = ['J', 'nabla_J']

#error_estimate_targets_outer = ['J', 'nabla_J']
#error_estimate_targets_inner = ['J', 'nabla_J', 'lin_J', 'nabla_lin_J']

#######################################################################

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
fig_1, ax_1 = plt.subplots(figsize=(6,4))
fig_2, ax_2 = plt.subplots(figsize=(6,4))
fig_3, ax_3 = plt.subplots(figsize=(6,4))

#cmap = plt.cm.get_cmap('viridis', 30) 
cmap = plt.cm.get_cmap('tab10')
colors = cmap.colors * 5   # repeat 5 times
cmap = ListedColormap(colors)

#######################################################################

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
 
        self.I = 0
        self.FOM_projector = SimpleBoundDomainProjector(
            model = FOM,
            bounds = FOM.bounds,
            reductor = None,
            use_sufficient_condition = False,        
            logger = self.logger
        )

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
                               projector: SimpleBoundDomainProjector = None,
                               alpha: float = 0.0,
                               use_error_estimator: bool = False) -> Tuple[NumpyVectorArray, float, bool, Dict]:

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
        
        u, u_dot = model.solve_state(q=current_q, 
                                     use_cached_operators=use_cached_operators,
                                     return_higher_orders=True)

        p, p_dot = model.solve_adjoint(q=current_q, 
                                       u=u, 
                                       use_cached_operators=use_cached_operators,
                                       return_higher_orders=True)

        current_J = model.objective(u=u,
                                    q=current_q,
                                    alpha=alpha)
        
        
        norm_d = model.compute_gradient_norm(previous_q - current_q)
        lhs =  previous_J - current_J
        rhs = kappa_arm / step_size * norm_d**2
        
        if abs(lhs) <= MACHINE_EPS:
            lhs = 0

        if abs(rhs) <= MACHINE_EPS:
            rhs = 0

        armijo_condition = lhs >= rhs
        if current_J > 0:
            errors = \
            self.estimate_errors(
                model = model,
                q_r = current_q,
                d_r = (current_q - previous_q),
                u_r = u,
                p_r = p,
                u_dot_r = u_dot,
                p_dot_r = p_dot,
                J_r = current_J,
                targets=error_estimate_targets_inner,
                use_cached_operators=use_cached_operators,
                use_error_estimator = use_error_estimator
            )
            abs_est_error_J_r = errors['err_J']
            J_rel_error = abs_est_error_J_r / current_J
        else:
            J_rel_error = np.inf
            
        TR_condition = J_rel_error <= eta
        condition = armijo_condition & TR_condition
        i += 1

        print("############")
        print(model.compute_gradient_norm(current_q-previous_q))
        print(step_size)
        print(previous_J)
        print(current_J)
        print(lhs)
        print(rhs)
        print(abs_est_error_J_r)
        print(eta)
        print(f"{J_rel_error:3.4e}")
        print(armijo_condition)
        print(TR_condition)

        while (not condition) and (i < max_iter):
            step_size = 0.5 * step_size
            
            if projector: 
                projector.pre_compute(center=previous_q)
                current_q = projector.project_domain(previous_q, step_size * search_direction)
            else:
                current_q = previous_q + step_size * search_direction

            u, u_dot = model.solve_state(q=current_q, 
                                         use_cached_operators=use_cached_operators, 
                                         return_higher_orders=True)
            p, p_dot = model.solve_adjoint(q=current_q, 
                                           u=u, 
                                           use_cached_operators=use_cached_operators, 
                                           return_higher_orders=True)

            current_J = model.objective(u=u,
                                        q=current_q,
                                        alpha=alpha)
            
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
                errors = \
                self.estimate_errors(
                    model = model,
                    q_r = current_q,
                    d_r = (current_q - previous_q),
                    u_r = u,
                    p_r = p,
                    u_dot_r = u_dot,
                    p_dot_r = p_dot,
                    J_r = current_J,
                    targets=error_estimate_targets_inner,
                    use_cached_operators=use_cached_operators,
                    use_error_estimator=use_error_estimator
                )                
                abs_est_error_J_r = errors['err_J']
                J_rel_error = abs_est_error_J_r / current_J
            else:
                J_rel_error = np.inf

            TR_condition = J_rel_error <= eta
            condition = armijo_condition & TR_condition

            print("############")
            print(model.compute_gradient_norm(current_q-previous_q))
            print(step_size)
            print(previous_J)
            print(current_J)
            print(lhs)
            print(rhs)
            print(abs_est_error_J_r)
            print(eta)
            print(f"{J_rel_error:3.4e}")
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

        return (current_q, current_J, model_unsufficent, TR_max_iter_cond, step_size, errors)

    def estimate_errors(self,
                        model: InstationaryModelIP,
                        q_r : VectorArray,
                        d_r : VectorArray = None,
                        u_r : VectorArray = None,
                        p_r : VectorArray = None,
                        u_dot_r : VectorArray = None,
                        p_dot_r : VectorArray = None,
                        lin_u_r : VectorArray = None,
                        lin_p_r : VectorArray = None,
                        J_r: float = None,
                        targets : str | List[str] = 'all',
                        use_error_estimator: bool = True,
                        use_cached_operators: bool = True) -> Dict:

        if id(model) == id(self.FOM):
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        if use_error_estimator:
            return self._estimate_error(
                model = model,
                q_r = q_r,
                d_r = d_r,
                u_r = u_r,
                p_r = p_r,
                u_dot_r = u_dot_r,
                p_dot_r = p_dot_r,
                lin_u_r = lin_u_r,
                lin_p_r = lin_p_r,
                J_r = J_r,
                targets = targets,
                use_cached_operators = use_cached_operators
            )        
        else:
            return self._calc_errors(
                model = model,
                q_r = q_r,
                d_r = d_r,
                u_r = u_r,
                p_r = p_r,
                lin_u_r = lin_u_r,
                lin_p_r = lin_p_r,
                targets = targets,
                use_cached_operators = use_cached_operators,
            )
    
    def _estimate_error(self,
                        model: InstationaryModelIP,
                        q_r : VectorArray,
                        d_r : VectorArray = None,
                        u_r : VectorArray = None,
                        p_r : VectorArray = None,
                        u_dot_r : VectorArray = None,
                        p_dot_r : VectorArray = None,
                        lin_u_r : VectorArray = None,
                        lin_p_r : VectorArray = None,
                        J_r: float = None,
                        targets : str | List[str] = 'all',
                        use_error_estimator: bool = True,
                        use_cached_operators: bool = True) -> Dict:

        ordered_targets = ['u', 'p', 'lin_u', 'lin_p', 'J', 'nabla_J', 'lin_J', 'nabla_lin_J']
        implemented_targets = ['J']

        if targets == 'all':
            targets = ordered_targets

        assert set(targets).issubset(implemented_targets)
        est_err_u = np.nan
        rel_est_err_u = np.nan
        est_err_p = np.nan
        rel_est_err_p = np.nan
        est_err_lin_u = np.nan
        rel_est_err_lin_u = np.nan
        est_err_lin_p = np.nan
        rel_est_err_lin_p = np.nan
        est_err_J = np.nan
        rel_est_err_J = np.nan
        est_err_nabla_J = np.nan
        rel_est_err_nabla_J = np.nan
        est_err_lin_J = np.nan
        rel_est_err_lin_J = np.nan
        est_err_nabla_lin_J = np.nan
        rel_est_err_nabla_lin_J = np.nan

        

        for target in targets:
            if target == 'J':
                assert q_r is not None
                assert u_r is not None
                assert u_dot_r is not None
                assert J_r is not None
                assert J_r > 0

                est_err_J = model.estimate_objective_error(
                    q = q_r,
                    u = u_r,
                    u_dot = u_dot_r,
                    J = J_r,
                    use_cached_operators = use_cached_operators,
                )

                self._logger.debug(f'Estimated err_J = {est_err_J:3.4e}')
                rel_est_err_J = est_err_J / J_r
                self._logger.debug(f'Estimated rel_err_J = {rel_est_err_J:3.4e}')
                continue
                
            raise ValueError

        return {
            'err_u' : est_err_u,
            'rel_est_err_u' : rel_est_err_u,
            'err_p' : est_err_p,
            'rel_est_err_p' : rel_est_err_p,
            'err_lin_u' : est_err_lin_u,
            'rel_est_err_lin_u' : rel_est_err_lin_u,
            'err_lin_p' : est_err_lin_p,
            'rel_est_err_lin_p' : rel_est_err_lin_p,
            'err_J' : est_err_J,
            'rel_est_err_J' : rel_est_err_J,
            'err_nabla_J' : est_err_nabla_J,
            'rel_est_err_nabla_J' : rel_est_err_nabla_J,
            'err_lin_J' : est_err_lin_J,
            'rel_est_err_lin_J' : rel_est_err_lin_J,
            'err_nabla_lin_J' : est_err_nabla_lin_J,
            'rel_est_err_nabla_lin_J' : rel_est_err_nabla_lin_J
        }

    def _calc_errors(self,
                     model: InstationaryModelIP,
                     q_r : VectorArray,
                     d_r : VectorArray = None,
                     u_r : VectorArray = None,
                     p_r : VectorArray = None,
                     lin_u_r : VectorArray = None,
                     lin_p_r : VectorArray = None,
                     targets : str | List[str] = 'all',
                     use_cached_operators: bool = True) -> Dict:

        
                
        ordered_targets = ['u', 'p', 'lin_u', 'lin_p', 'J', 'nabla_J', 'lin_J', 'nabla_lin_J']

        if targets == 'all':
            targets = ordered_targets

        
        assert set(targets).issubset(
            ['u', 'p', 'lin_u', 'lin_p', 'J', 'nabla_J', 'lin_J', 'nabla_lin_J']
        )
    
        if set(targets).issubset(['lin_u', 'lin_p', 'lin_J', 'nabla_lin_J']):
            assert d_r is not None

        err_u = np.nan
        rel_err_u = np.nan
        err_p = np.nan
        rel_err_p = np.nan
        err_lin_u = np.nan
        rel_err_lin_u = np.nan
        err_lin_p = np.nan
        rel_err_lin_p = np.nan
        err_J = np.nan
        rel_err_J = np.nan
        err_nabla_J = np.nan
        rel_err_nabla_J = np.nan
        err_lin_J = np.nan
        rel_err_lin_J = np.nan
        err_nabla_lin_J = np.nan
        rel_err_nabla_lin_J = np.nan

        q = self.reductor.reconstruct(q_r, basis='parameter_basis')
        if d_r is not None:
            d = self.reductor.reconstruct(d_r, basis='parameter_basis')

        u = None
        p = None
        lin_u = None
        lin_p = None
        J = None
        nabla_J = None
        lin_J = None
        nabla_lin_J = None

        J_r = None
        nabla_J_r = None
        lin_J_r = None
        nabla_lin_J_r = None

        required_quantities = []

        for target in targets:
            if target == 'u':
                required_quantities += ['u']
            elif target == 'p':
                required_quantities += ['u', 'p']
            elif target == 'lin_u':
                required_quantities += ['u', 'lin_u']
            elif target == 'lin_p':
                required_quantities += ['lin_u', 'lin_p']
            elif target == 'J':
                required_quantities += ['u', 'J']
            elif target == 'nabla_J':
                required_quantities += ['u', 'p', 'nabla_J']
            elif target == 'lin_J':
                required_quantities += ['lin_u', 'lin_J']
            elif target == 'nabla_lin_J':
                required_quantities += ['u', 'lin_p', 'nabla_lin_J']

        required_quantities = [item for item in ordered_targets if item in required_quantities]

        for required_quantity in required_quantities:
            if required_quantity == 'u':
                if u_r is None:
                    u_r = model.solve_state(q_r, use_cached_operators=use_cached_operators)
                
                _u_r = self.reductor.reconstruct(u_r, basis='state_basis')
                u = self.FOM.solve_state(q, use_cached_operators=use_cached_operators)


                diff = u - _u_r
                err_u = np.sqrt(self.FOM.products['bochner_prod_V'].apply2(diff, diff))[0,0]
                self._logger.debug(f'Actual err_u = {err_u:3.4e}')

                norm_u = np.sqrt(self.FOM.products['bochner_prod_V'].apply2(u, u))[0,0]
                rel_err_u = err_u / norm_u
                self._logger.debug(f'Actual rel_err_u = {rel_err_u:3.4e}')
                continue

            if required_quantity == 'p':
                if p_r is None:
                    p_r = model.solve_adjoint(q_r, u_r, use_cached_operators=use_cached_operators)
                
                _p_r = self.reductor.reconstruct(p_r, basis='state_basis')
                p = self.FOM.solve_adjoint(q, u, use_cached_operators=use_cached_operators)
                
                diff = p - _p_r
                err_p = np.sqrt(self.FOM.products['bochner_prod_V'].apply2(diff, diff))[0,0]
                self._logger.debug(f'Actual err_p = {err_p:3.4e}')

                norm_p = np.sqrt(self.FOM.products['bochner_prod_V'].apply2(p, p))[0,0]
                rel_err_p = err_p / norm_p
                self._logger.debug(f'Actual rel_err_p = {rel_err_p:3.4e}')
                continue
            
            if required_quantity == 'lin_u':
                if lin_u_r is None:
                    lin_u_r = model.solve_linearized_state(q_r, d_r, u_r, use_cached_operators=use_cached_operators)
                
                _lin_u_r = self.reductor.reconstruct(lin_u_r, basis='state_basis')
                lin_u = self.FOM.solve_linearized_state(q, d, u, use_cached_operators=use_cached_operators)

                diff = lin_u - _lin_u_r
                err_lin_u = np.sqrt(self.FOM.products['bochner_prod_V'].apply2(diff, diff))[0,0]
                self._logger.debug(f'Actual err_lin_u = {err_lin_u:3.4e}')
                
                norm_lin_u = np.sqrt(self.FOM.products['bochner_prod_V'].apply2(lin_u, lin_u))[0,0]
                rel_err_lin_u = err_lin_u / norm_lin_u
                self._logger.debug(f'Actual rel_err_lin_u = {rel_err_lin_u:3.4e}')
                continue

            if required_quantity == 'lin_p':
                if lin_p_r is None:
                    lin_p_r = model.solve_linearized_adjoint(q_r, u_r, lin_u_r, use_cached_operators=use_cached_operators)
                
                _lin_p_r = self.reductor.reconstruct(lin_u_r, basis='state_basis')
                lin_p = self.FOM.solve_linearized_adjoint(q, u, lin_u, use_cached_operators=use_cached_operators)

                diff = lin_p - _lin_p_r
                err_lin_p = np.sqrt(self.FOM.products['bochner_prod_V'].apply2(diff, diff))[0,0]
                self._logger.debug(f'Actual err_lin_p = {err_lin_p:3.4e}')

                norm_lin_p = np.sqrt(self.FOM.products['bochner_prod_V'].apply2(lin_p, lin_p))[0,0]
                rel_err_lin_p = err_lin_p / norm_lin_p
                self._logger.debug(f'Actual rel_err_lin_p = {rel_err_lin_p:3.4e}')
                continue
            
            if required_quantity == 'J':
                if J_r is None:
                    J_r = model.objective(u_r, q_r)

                J = self.FOM.objective(u, q)
                err_J = np.abs(J - J_r)
                self._logger.debug(f'Actual err_J = {err_J:3.4e}')

                rel_err_J = err_J / np.abs(J)
                self._logger.debug(f'Actual rel_err_J = {rel_err_J:3.4e}')
                continue


            if required_quantity == 'nabla_J':
                if nabla_J_r is None:
                    nabla_J_r = model.gradient(u_r, p_r, q_r, use_cached_operators=use_cached_operators)
                
                _nabla_J_r = self.reductor.reconstruct(nabla_J_r, basis='parameter_basis')
                nabla_J = self.FOM.gradient(u, p, q, use_cached_operators=use_cached_operators)
                
                diff = nabla_J - _nabla_J_r
                if self.FOM.q_time_dep:
                    err_nabla_J = np.sqrt(self.FOM.products['bochner_prod_Q'].apply2(diff, diff))[0,0]
                    norm_nabla_J = np.sqrt(self.FOM.products['bochner_prod_Q'].apply2(nabla_J, nabla_J))[0,0]
                else:
                    err_nabla_J = np.sqrt(self.FOM.products['prod_Q'].apply2(diff, diff))[0,0]
                    norm_nabla_J = np.sqrt(self.FOM.products['prod_Q'].apply2(nabla_J, nabla_J))[0,0]

                self._logger.debug(f'Actual err_nabla_J = {err_nabla_J:3.4e}')
                rel_err_nabla_J = err_nabla_J / norm_nabla_J
                self._logger.debug(f'Actual rel_err_nabla_J = {rel_err_nabla_J:3.4e}')
                continue

            if required_quantity == 'lin_J':
                if lin_J_r is None:
                    lin_J_r = model.linearized_objective(q_r, d_r, u_r, lin_u_r, 0.0, use_cached_operators=use_cached_operators)
                
                lin_J = self.FOM.linearized_objective(q, d, u, lin_u, 0.0, use_cached_operators=use_cached_operators)
                err_lin_J = np.abs(lin_J - lin_J_r)
                self._logger.debug(f'Actual err_lin_J = {err_lin_J:3.4e}')

                rel_err_lin_J = err_lin_J / np.abs(lin_J)
                self._logger.debug(f'Actual rel_err_lin_J = {rel_err_lin_J:3.4e}')
                continue
  
            if required_quantity == 'nabla_lin_J':
                if nabla_lin_J_r is None:
                    nabla_lin_J_r = model.linearized_gradient(q_r, d_r, u_r, lin_p_r, 0.0, use_cached_operators=use_cached_operators)
                
                _nabla_lin_J_r = self.reductor.reconstruct(nabla_lin_J_r, basis='parameter_basis')
                nabla_lin_J = self.FOM.linearized_gradient(q, d, u, lin_u, 0.0, use_cached_operators=use_cached_operators)

                diff = nabla_lin_J - _nabla_lin_J_r
                if self.FOM.q_time_dep:
                    err_nabla_lin_J = np.sqrt(self.FOM.products['bochner_prod_Q'].apply2(diff, diff))[0,0]
                    norm_nabla_lin_J = np.sqrt(self.FOM.products['bochner_prod_Q'].apply2(nabla_lin_J, nabla_lin_J))[0,0]
                else:
                    err_nabla_lin_J = np.sqrt(self.FOM.products['prod_Q'].apply2(diff, diff))[0,0]
                    norm_nabla_lin_J = np.sqrt(self.FOM.products['prod_Q'].apply2(nabla_lin_J, nabla_lin_J))[0,0]

                self._logger.debug(f'Actual err_nabla_lin_J = {err_nabla_lin_J:3.4e}')
                rel_err_nabla_lin_J = err_nabla_lin_J / norm_nabla_lin_J
                self._logger.debug(f'Actual rel_err_nabla_lin_J = {rel_err_nabla_lin_J:3.4e}')
                continue
                    
            raise ValueError
        
        return {
            'err_u' : err_u,
            'rel_err_u' : rel_err_u,
            'err_p' : err_p,
            'rel_err_p' : rel_err_p,
            'err_lin_u' : err_lin_u,
            'rel_err_lin_u' : rel_err_lin_u,
            'err_lin_p' : err_lin_p,
            'rel_err_lin_p' : rel_err_lin_p,
            'err_J' : err_J,
            'rel_err_J' : rel_err_J,
            'err_nabla_J' : err_nabla_J,
            'rel_err_nabla_J' : rel_err_nabla_J,
            'err_lin_J' : err_lin_J,
            'rel_err_lin_J' : rel_err_lin_J,
            'err_nabla_lin_J' : err_nabla_lin_J,
            'rel_err_nabla_lin_J' : rel_err_nabla_lin_J
        }
    
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
              TR_enforcement: str | None = None,
              lin_solver_parms: Dict = None,
              TR_params: Dict = None,
              use_cached_operators: bool = False,
              dump_IRGNM_intermed_stats: bool = False,
              dump_every_nth_loop: int = 0,
              projector: SimpleBoundDomainProjector = None,
              use_error_estimator: bool = False) -> Tuple[VectorArray, Dict]: 

        assert q_0 in model.Q
        assert tol > 0
        assert tau > 0
        assert noise_level >= 0
        #assert 0 < theta < Theta < 1

        assert lin_solver_parms is not None        

        if TR_enforcement is not None:
            assert TR_enforcement in ['check_error', 'backtracking']
            assert TR_params is not None
            method_name = 'TR-IRGNM'
        else:
            method_name = 'IRGNM'

        self.IRGNM_statistics = {
            'IRGNM_idx' : self.IRGNM_idx,
            "q" : [],
            "alpha" : [],
            "J" : [],
            "norm_nabla_J" : [],
            "total_runtime" : [],
            "stagnation_flag" : False,
            "FOM_num_calls" : {},
            "counts" : {},
            "errors" : {
                'err_u' : [],
                'rel_err_u' : [],
                'err_p' : [],
                'rel_err_p' : [],
                'err_lin_u' : [],
                'rel_err_lin_u' : [],
                'err_lin_p' : [],
                'rel_err_lin_p' : [],
                'err_J' : [],
                'rel_err_J' : [],
                'err_nabla_J' : [],
                'rel_err_nabla_J' : [],
                'err_lin_J' : [],
                'rel_err_lin_J' : [],
                'err_nabla_lin_J' : [],
                'rel_err_nabla_lin_J' : []
            }
        }
        
        counts = {
            'IRGNM_loop_iter' : -1,
            'reg_loop_iter' : [],
            'lin_solver_iter' : [],
            'loop_terminated' : []
        }

        stagnation_flag = False
        
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
        self.IRGNM_statistics["total_runtime"].append(timer() - start_time)

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


        loop_terminated = False

        x = np.zeros(shape=(model.V.dim,))
        y = np.zeros(shape=(model.V.dim,))

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

            # u_prev = self.FOM.solve_state(
            #     q = self.FOM_projector.project_domain(center=
            #         self.reductor.reconstruct(q, basis='parameter_basis')
            #     )
            # )
            
            TR_max_iter_cond = False
            model_unsufficent = False

            if TR_enforcement == 'backtracking':
                self.logger.info(f"Enforcing TR condition using 'backtracking'.")
                q_TR, _, model_unsufficent, TR_max_iter_cond, step_size, errors = self._armijo_TR_line_serach(
                    model = model,
                    previous_q = q,
                    previous_J = J,
                    search_direction = d,
                    **TR_params,
                    use_cached_operators=use_cached_operators,
                    projector=projector,
                    use_error_estimator=use_error_estimator
                )

                if TR_max_iter_cond:
                    break

                q = q_TR

            elif TR_enforcement == 'check_error':
                self.logger.info(f"Enforcing TR condition using 'check_error'.")

                if projector:
                    projector.pre_compute(center=q)
                    next_q = projector.project_domain(q, d)
                else:
                    next_q = q + d

                u, u_dot = model.solve_state(q=next_q, use_cached_operators=use_cached_operators)
                p, p_dot = model.solve_adjoint(q=next_q, u=u, use_cached_operators=use_cached_operators)
                next_J = model.objective(u=u,q=next_q)

                if next_J > 0:
                    errors = \
                    self.estimate_errors(
                        model = model,
                        q_r = next_q,
                        d_r = d,
                        u_r = u,
                        p_r = p,
                        u_dot_r = u_dot,
                        p_dot_r = p_dot,
                        J_r = J,
                        targets = error_estimate_targets_inner,
                        use_cached_operators=use_cached_operators,
                        use_error_estimator=use_error_estimator
                    )
                    abs_est_error_J_r = errors['err_J']
                    J_rel_error = abs_est_error_J_r / next_J
                else:
                    J_rel_error = np.inf

                eta = TR_params['eta']
                beta = TR_params['beta']

                if J_rel_error <= eta:
                    q = next_q

                if (J_rel_error > beta * eta):
                    model_unsufficent = True

                print("############")
                print(next_J)
                print(abs_est_error_J_r)
                print(eta)
                print(f"{J_rel_error:3.4e}")
                print(J_rel_error <= eta)
                print(J_rel_error <= beta * eta)

            else:
                q += d
                # projector.pre_compute(center=q)
                # next_q = projector.project_domain(q, d)
            
            # _u = self.FOM.solve_state(q = self.reductor.reconstruct(q, basis='parameter_basis'))
            # _p = self.FOM.solve_adjoint(q = self.reductor.reconstruct(q, basis='parameter_basis'), u = _u)

            # u_np = _u.to_numpy()          # shape: (nt, ndofs)
            # nt, ndofs = u_np.shape

            # dof_index = int(1e4)                     # choose any DOF index in [0, ndofs)
            # signal = u_np[:, dof_index]       # shape: (nt,)

            # signal = signal - np.mean(signal)

            # dt = 0.01                         # <-- set this to your actual timestep
            # U_fft = np.fft.rfft(signal)       # complex spectrum, non-negative freqs
            # freqs = np.fft.rfftfreq(nt, d=dt) # frequency axis (Hz)

            # ax_3.plot(freqs, np.abs(U_fft))
            # ax_3.grid(True)
            # fig_3.savefig(self.save_path / "fourier_u.pdf")


            # u_r = self.reductor.reconstruct(u, basis='state_basis')
            # p_r = self.reductor.reconstruct(p, basis='state_basis')
            

            # self.I += 1
            # self.FOM.A.material_model.save_time_series(
            #     [v.real_part.impl for v in u_r.vectors],
            #     str(f'u_r_{self.I}'),
            #     str(self.save_path),
            #     np.linspace(self.FOM.T_initial, self.FOM.T_final, self.FOM.nt+1)
            # )

            # diff = _u - u_r
            # self.FOM.A.material_model.save_time_series(
            #     [v.real_part.impl for v in diff.vectors],
            #     str(f'diff_u_{self.I}'),
            #     str(self.save_path),
            #     np.linspace(self.FOM.T_initial, self.FOM.T_final, self.FOM.nt+1)
            # )

            # self.FOM.A.material_model.save_time_series(
            #     [v.real_part.impl for v in p_r.vectors],
            #     str(f'p_r_{self.I}'),
            #     str(self.save_path),
            #     np.linspace(self.FOM.T_initial, self.FOM.T_final, self.FOM.nt+1)
            # )

            # diff = _p -p_r
            # self.FOM.A.material_model.save_time_series(
            #     [v.real_part.impl for v in diff.vectors],
            #     str(f'diff_p_{self.I}'),
            #     str(self.save_path),
            #     np.linspace(self.FOM.T_initial, self.FOM.T_final, self.FOM.nt+1)
            # )


     


            # basis = 'state_basis'
            # _basis = self.reductor.bases[basis]
            
            # coeff_u = np.sum((_u.inner(_basis, self.reductor.products[basis]))**2, axis=0)
            # err_i_u = np.sum(self.reductor.products[basis].pairwise_apply2(_u,_u)) - np.cumsum(coeff_u)
            
            # coeff_p = np.sum((_p.inner(_basis, self.reductor.products[basis]))**2, axis=0)
            # err_i_p = np.sum(self.reductor.products[basis].pairwise_apply2(_p,_p)) - np.cumsum(coeff_p)

            # self.I += 1
            # #color = cmap(self.I)
            # color = cmap(i)
            # ax_1.semilogy(err_i_u, color=color)
            # ax_1.semilogy(err_i_p, color=color, linestyle="--")
            # ax_1.set_ylim([1e-8, 1e3])
            # ax_1.grid(True)

            # fig_1.savefig(self.save_path / "inner_plot.pdf")


            # _u_r = model.solve_state(q)
            # u_r = self.reductor.reconstruct(_u_r, basis='state_basis')

            #self.I += 1
            # diff_u = _u - u_r
            # self.FOM.A.material_model.save_time_series(
            #     [v.real_part.impl for v in diff_u.vectors],
            #     str(f'diff_u_{self.I}'),
            #     str(self.save_path),
            #     np.linspace(0, len(_basis), len(_basis))
            # )

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

            for key in self.IRGNM_statistics["errors"].keys():
                if TR_enforcement is not None:
                    self.IRGNM_statistics["errors"][key].append(errors[key])
                else:
                    self.IRGNM_statistics["errors"][key].append(np.nan)

            #stagnation check
            if i > 3:
                buffer = self.IRGNM_statistics["J"][-3:]
                if abs(buffer[0] - buffer[1]) / abs(buffer[0]) < STAGNATION_TOL and abs(buffer[1] - buffer[2]) / abs(buffer[1])< STAGNATION_TOL:
                    self.IRGNM_statistics["stagnation_flag"] = True
                    self.logger.info(f"Stop at iteration {i+1} of {int(i_max)}, due to stagnation.")
                    stagnation_flag = True
                    break

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
        
        # u = self.FOM.solve_state(
        #     q = self.FOM_projector.project_domain(center=
        #         self.reductor.reconstruct(q, basis='parameter_basis')
        #     )
        # )
        # u_r = self.reductor.reconstruct(model.solve_state(q), basis='state_basis')


        # self.I += 1
        # diff = u - u_r
        # self.FOM.A.material_model.save_time_series(
        #     [v.real_part.impl for v in diff.vectors],
        #     str(f'diff_u_{self.I}_last'),
        #     str(self.save_path),
        #     np.linspace(self.FOM.T_initial, self.FOM.T_final, self.FOM.nt+1)
        # )

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
                                        projector=self.FOM_projector)

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
        self.parameter_snapshots = self.FOM.Q.empty()
        self.parameter_snapshots.append(nabla_J)
        self.parameter_snapshots.append(q)
        self.parameter_snapshots.append(self.FOM.Q.make_array(self.FOM.setup['q_circ']))

        if self.FOM.setup['q_time_dep']:
            self.logger.debug(f"Performing HaPOD on parameter snapshots.")
            _parameter_snapshots, _ = self._HaPOD(snapshots=self.parameter_snapshots, 
                                                  basis='parameter_basis',
                                                  product=self.FOM.products['prod_Q'])

        self.reductor.extend_basis(
             U = _parameter_snapshots,
             basis = 'parameter_basis'
        )
        self.QrFOM = self.reductor.reduce()
        IRGNM_statistic = {}

        while np.sqrt(2 * J) >= tol+tau*noise_level and i<i_max:
            self.logger.info(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            self.logger.warning(f"Qr-IRGNM iteration {i}: J = {J:3.4e} is not sufficent: {np.sqrt(2 * J):3.4e} > {(tol+tau*noise_level):3.4e}.")
            self.logger.info(f'Start Qr-IRGNM iteration {i}: J = {J:3.4e}, norm_nabla_J = {norm_nabla_J:3.4e}, alpha = {alpha:1.4e}')
            self.logger.info(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            q_r = self.reductor.project_vectorarray(q, 'parameter_basis')
            q_r = self.QrFOM.Q.make_array(q_r)
            
            self.IRGNM_idx += 1
            
            inner_loop_start_time = timer()
            q_r, IRGNM_statistic_ = self.IRGNM(model = self.QrFOM,
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
                                               use_cached_operators = use_cached_operators,
                                               use_error_estimator=False)
            
            q = self.reductor.reconstruct(q_r, basis='parameter_basis')
            assert self.FOM_projector.project_domain(center=q) == q
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
            self.parameter_snapshots = self.FOM.Q.empty()
            self.parameter_snapshots.append(nabla_J)
            
            if self.FOM.q_time_dep:
                self.logger.debug(f"Performing HaPOD on parameter snapshots.")
                _parameter_snapshots, _ = self._HaPOD(snapshots=self.parameter_snapshots, 
                                                      basis='parameter_basis',
                                                      product=self.FOM.products['prod_Q'])

            self.reductor.extend_basis(
                U = _parameter_snapshots,
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
            optimizer_parameter['error_estimator_types'],
            use_adjoint_space = optimizer_parameter["use_adjoint_space"],
            parallel = optimizer_parameter["offline_parallel"]
        )
        self.snapshot_preprocessor = SnapshotPreprocessor(
            FOM = FOM
        )

        if optimizer_parameter["use_adjoint_space"]:
            self.reduced_bases = ['parameter_basis','state_basis', 'adjoint_basis']
        else:
            self.reduced_bases = ['parameter_basis','state_basis']
        
        # if not optimizer_parameter['enrichment']['parameter_basis']['reduced_basis']:
        #     self.reduced_bases = self.reduced_bases[1:]


        self.QrVrROM = None

        self.snapshots = {
            'parameter_basis' : FOM.Q.empty(),
            'state_basis' : FOM.V.empty(),
            'adjoint_basis' : FOM.V.empty(),
        }

        self.statistics = {
            "q" : [],
            "eta" : [],
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
                'extend_runtime' : {    
                    'parameter_basis' : [],
                    'state_basis' : [],
                    'adjoint_basis' : [],
                },  
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

    def extend_bases_and_rebuild_QrVrROM(self,
                                         bases: List[str],
                                         enrichment : Dict,
                                         i: int = -1) -> InstationaryModelIP:
        
        for basis in bases:
            extend_start_time = timer()

            assert basis in ['parameter_basis', 'state_basis', 'adjoint_basis']
            assert enrichment[basis]

            self.logger.debug(f"Extending '{basis}'")
            snapshots = self.snapshots[basis]

            # self.statistics['extention_stats']['snapshot_projection_error'][basis].append(
            #     self.reductor.calc_projection_error(
            #         x = snapshots,
            #         basis = basis,
            #         normalize = False
            #     )
            # )

            snapshots = self.snapshot_preprocessor.preprocess(
                snapshots = snapshots,
                product = self.reductor.products[basis],
                config = enrichment[basis]['compression']
            )
            
            try:
                self.reductor.extend_basis(
                    U = snapshots,
                    basis = basis
                )
                
            except ExtensionError:
                self._logger.warning(f"No new vectors were added to {basis}.")    


            self.statistics["outer_loop_runtime"]['extend_runtime'][basis][-1] += (timer() - extend_start_time)

            
        self.reductor.dims_history['parameter_basis'].append(self.reductor.get_bases_dim('parameter_basis'))
        self.reductor.dims_history['state_basis'].append(self.reductor.get_bases_dim('state_basis'))
        self.reductor.dims_history['adjoint_basis'].append(self.reductor.get_bases_dim('adjoint_basis'))
        
        self.logger.debug(f"Dim Qr-space = {self.reductor.get_bases_dim('parameter_basis')}")
        self.logger.debug(f"Dim Vr-space = {self.reductor.get_bases_dim('state_basis')}")
        self.logger.debug(f"Dim Wr-space = {self.reductor.get_bases_dim('adjoint_basis')}")


        reduce_start_time = timer()
        self.logger.debug(f"Creating Qr-Vr-ROM")
        QrVrROM = self.reductor.reduce() 
        print("-------------------------------")
        print(self.statistics["outer_loop_runtime"]['reduce_runtime'][-1])
        print(timer() - reduce_start_time)
        self.statistics["outer_loop_runtime"]['reduce_runtime'][-1] += (timer() - reduce_start_time)
        
        return QrVrROM    
    
    def _reset_snapshots(self) -> None:
        self.snapshots = {
            'parameter_basis' : self.FOM.Q.empty(),
            'state_basis' : self.FOM.V.empty(),
            'adjoint_basis' : self.FOM.V.empty(),
        }
       
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

        use_error_estimator = self.optimizer_parameter["use_error_estimator"]
        use_adjoint_space = self.optimizer_parameter["use_adjoint_space"]
        offline_parallel = self.optimizer_parameter["offline_parallel"]
        reg_AGC_step = self.optimizer_parameter["reg_AGC_step"]
        TR_enforcement = self.optimizer_parameter["TR_enforcement"]
        assert TR_enforcement in ['check_error', 'backtracking']

        lin_solver_parms = self.optimizer_parameter['lin_solver_parms']
        use_cached_operators = self.optimizer_parameter['use_cached_operators']        
        enrichment = self.optimizer_parameter['enrichment']

        dump_every_nth_loop = self.optimizer_parameter['dump_every_nth_loop']

        eta0 = self.optimizer_parameter["eta0"]
        eta_min = self.optimizer_parameter["eta_min"]
        eta_max = self.optimizer_parameter["eta_max"]
        kappa_arm = self.optimizer_parameter["kappa_arm"]
        beta_1 = self.optimizer_parameter["beta_1"]
        beta_2 = self.optimizer_parameter["beta_2"]
        beta_3 = self.optimizer_parameter["beta_3"]

        lin_u = None
        lin_p = None
        nabla_lin_J = None
        time_step_nabla_J = None


        start_time = timer()
        i = 0
        alpha = alpha_0
        delta = noise_level

        solve_snapshot_FOM_start_time = timer()
        q = self.FOM.Q.make_array(q_0)
        u = self.FOM.solve_state(q, use_cached_operators=False)        
        p = self.FOM.solve_adjoint(q, u, use_cached_operators=False)
        J = self.FOM.objective(u)
        nabla_J, time_step_nabla_J = self.FOM.gradient(u, 
                                                       p, 
                                                       q, 
                                                       use_cached_operators=use_cached_operators,
                                                       return_per_time_step = True)
        
        if enrichment['state_basis']['additional_snapshots']['include_lins'] or enrichment['parameter_basis']['additional_snapshots']['include_lin_grad']:
            direction = -nabla_J
            lin_u = self.FOM.solve_linearized_state(q, direction, u, use_cached_operators=use_cached_operators)
            lin_p = self.FOM.solve_linearized_adjoint(q, u, lin_u, use_cached_operators=use_cached_operators)

        if enrichment['parameter_basis']['additional_snapshots']['include_lin_grad']:
            nabla_lin_J = self.FOM.linearized_gradient(q, nabla_J, u, lin_p, alpha=0, use_cached_operators=use_cached_operators)
        
        norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)
        self.statistics['outer_loop_runtime']['solve_snapshot_FOM_runtime'].append(timer()  - solve_snapshot_FOM_start_time)
        for basis in self.reduced_bases:
            self.statistics["outer_loop_runtime"]['extend_runtime'][basis].append(0.0)
        self.statistics["outer_loop_runtime"]['reduce_runtime'].append(0.0)

        #assert norm_nabla_J > 0

        inital_agc_armijo_step_size = 0.5 / norm_nabla_J
        inital_agc_armijo_step_size = np.min([inital_agc_armijo_step_size, 1e-3])
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
        self.logger.debug(f"  use_error_estimator : {use_error_estimator}")
        self.logger.debug(f"  use_adjoint_space : {use_adjoint_space}")        
        self.logger.debug(f"  offline_parallel : {offline_parallel}")
        self.logger.debug(f"  reg_AGC_step : {reg_AGC_step}")
        self.logger.debug(f"  TR_enforcement : {TR_enforcement}")
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
        self.logger.debug(f"  eta_min : {eta_min:3.4e}")
        self.logger.debug(f"  eta_max : {eta_max:3.4e}")
        self.logger.debug(f"  kappa_arm : {kappa_arm:3.4e}")
        self.logger.debug(f"  beta_1 : {beta_1:3.4e}")
        self.logger.debug(f"  beta_2 : {beta_2:3.4e}")
        self.logger.debug(f"  beta_3 : {beta_3:3.4e}")


        self._reset_snapshots()
        additional_parameter_snapshots, additional_state_snapshots, additional_adjoint_snapshots = \
        self.snapshot_preprocessor.additional_snapshots(
            config = enrichment,
            bases = self.reduced_bases,
            q = q,
            u = u,
            lin_u = lin_u,
            lin_p = lin_p,
            nabla_J = nabla_J,
            nabla_lin_J = nabla_lin_J,
            time_step_nabla_J = time_step_nabla_J,
            use_cached_operators = use_cached_operators,
        )

        self.logger.debug(f"Extending Qr-snapshots")
        if not enrichment['parameter_basis']['reduced_basis']:
            self.logger.debug('Using reduced parameter space')

            assert isinstance(self.FOM.Q, NumpyVectorSpace)
            self.snapshots['parameter_basis'].append( 
                self.FOM.Q.make_array(np.identity(self.FOM.Q.dim))
            )
        else:
            self.snapshots['parameter_basis'].append(nabla_J)
            self.snapshots['parameter_basis'].append(q)
            self.snapshots['parameter_basis'].append(self.FOM.Q.make_array(self.FOM.setup['q_circ']))
            self.snapshots['parameter_basis'].append(
                additional_parameter_snapshots
            )

        self.logger.debug(f"Extending Vr-snapshots")

        if self.reductor.use_adjoint_space:
            self.snapshots['state_basis'].append(u)
            self.snapshots['adjoint_basis'].append(p)
            
            self.snapshots['adjoint_basis'].append(
                additional_adjoint_snapshots
            )
        else:
            self.snapshots['state_basis'].append(u)
            self.snapshots['state_basis'].append(p)

        self.snapshots['state_basis'].append(
            additional_state_snapshots
        )

        # self.snapshots['state_basis'].append(self.FOM.bilinear_cost_term.apply(self.snapshots['state_basis']))
        # self.snapshots['state_basis'].append(self.FOM.linear_cost_term)

        # print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")
        # print(len(self.snapshots['state_basis']))
    
                
        self.QrVrROM = self.extend_bases_and_rebuild_QrVrROM(
            bases=self.reduced_bases,
            enrichment=enrichment, 
            i = i
        )

        ############################################################

        # basis = 'state_basis'
        # _basis = self.reductor.bases[basis]

        # coeff_u = np.sum((u.inner(_basis, self.reductor.products[basis]))**2, axis=0)
        # err_i_u = np.sum(self.reductor.products[basis].pairwise_apply2(u,u)) - np.cumsum(coeff_u)
        
        # coeff_p = np.sum((p.inner(_basis, self.reductor.products[basis]))**2, axis=0)
        # err_i_p = np.sum(self.reductor.products[basis].pairwise_apply2(p,p)) - np.cumsum(coeff_p)

        # # err_i_u = err_i_u[err_i_u > 0]
        # # err_i_p = err_i_p[err_i_p > 0]

        # color = cmap(i)
        # ax_2.semilogy(err_i_u, color=color)
        # ax_2.semilogy(err_i_p, color=color, linestyle="--")
        # ax_2.set_ylim([1e-18, 1e3])
        # ax_2.grid(True)
        
        # fig_2.savefig(self.save_path / "coeffs_after_enrich.pdf")

        ############################################################


        q_r = self.reductor.project_vectorarray(q, 'parameter_basis')
        q_r = self.QrVrROM.Q.make_array(q_r)

        u_r, u_dot_r = self.QrVrROM.solve_state(q_r,
                                                use_cached_operators = use_cached_operators,
                                                return_higher_orders = True)
        p_r, p_dot_r = self.QrVrROM.solve_adjoint(q_r, 
                                                  u_r,
                                                  use_cached_operators = use_cached_operators,
                                                  return_higher_orders = True)
        J_r = self.QrVrROM.objective(u_r)
        nabla_J_r = self.QrVrROM.gradient(u_r, p_r, q_r)
        norm_nabla_J_r = self.QrVrROM.compute_gradient_norm(nabla_J_r)

        # p_r_start = self.reductor.reconstruct(p_r, basis='state_basis')
        # self.FOM.A.material_model.save_time_series(
        #     [v.real_part.impl for v in p_r_start.vectors],
        #     str('p_r_start'),
        #     str(self.save_path),
        #     np.linspace(self.FOM.T_initial, self.FOM.T_final, self.FOM.nt+1)
        # )

        # lin_u_r = self.QrVrROM.solve_linearized_state(q_r, nabla_J_r, u_r, use_cached_operators=use_cached_operators)
        # lin_p_r = self.QrVrROM.solve_linearized_adjoint(q_r, u_r, lin_u_r, use_cached_operators=use_cached_operators)

        # lin_p_r_start = self.reductor.reconstruct(lin_p_r, basis='state_basis')
        # self.FOM.A.material_model.save_time_series(
        #     [v.real_part.impl for v in lin_p_r_start.vectors],
        #     str('lin_p_r_start'),
        #     str(self.save_path),
        #     np.linspace(self.FOM.T_initial, self.FOM.T_final, self.FOM.nt+1)
        # )

        # self.FOM.A.material_model.save_time_series(
        #     [v.real_part.impl for v in lin_p.vectors],
        #     str('lin_p'),
        #     str(self.save_path),
        #     np.linspace(self.FOM.T_initial, self.FOM.T_final, self.FOM.nt+1)
        # )

        errors = \
        self.estimate_errors(
            model=self.QrVrROM,
            q_r = q_r,
            u_r = u_r,
            p_r = p_r,
            u_dot_r = u_dot_r,
            p_dot_r = p_dot_r,
            J_r = J,
            targets = error_estimate_targets_outer,
            use_cached_operators=use_cached_operators,
            use_error_estimator=use_error_estimator
        )

        if J_r > 0:
            abs_est_error_J_r = errors['err_J']
            rel_est_error_J_r = abs_est_error_J_r / J_r
        else:
            rel_est_error_J_r = np.inf

        if norm_nabla_J_r > 0:
            abs_est_error_nabla_J_r = errors['err_nabla_J']
            rel_est_error_nabla_J_r = abs_est_error_nabla_J_r / norm_nabla_J_r
        else:
            rel_est_error_nabla_J_r = np.inf

        self.statistics["q"].append(q)
        self.statistics["eta"].append(eta)
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
        last_inner_alpha = None
        IRGNM_statistics = {}

        # u_r = self.reductor.reconstruct(u_r, basis='state_basis')
        # print(u_r.to_numpy())
        # self.FOM.A.material_model.save_time_series(
        #     [v.real_part.impl for v in u_r.vectors],
        #     str('u_r'),
        #     str(self.save_path),
        #     np.linspace(self.FOM.T_initial, self.FOM.T_final, self.FOM.nt+1)
        # )

        # if self.QrVrROM.use_adjoint_space:
        #     _basis = 'adjoint_basis'
        # else:
        #     _basis = 'state_basis'

        # print("AAAAAAAAAAAAAAAAAAAa")
        # p_r = self.reductor.reconstruct(p_r, basis=_basis)
        # print(p_r.to_numpy())
        # self.FOM.A.material_model.save_time_series(
        #     [v.real_part.impl for v in p_r.vectors],
        #     str('p_r'),
        #     str(self.save_path),
        #     np.linspace(self.FOM.T_initial, self.FOM.T_final, self.FOM.nt+1)
        # )

        
        # lin_u_r = self.QrVrROM.solve_linearized_state(q_r, nabla_J_r, u_r, use_cached_operators=use_cached_operators)
        # lin_p_r = self.QrVrROM.solve_linearized_adjoint(q_r, u_r, lin_u_r, use_cached_operators=use_cached_operators)
        # if self.QrVrROM.use_adjoint_space:
        #     _basis = 'adjoint_basis'
        # else:
        #     _basis = 'state_basis'

        # print(self.reductor.reconstruct(u_r, basis='state_basis').to_numpy())
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(self.reductor.reconstruct(lin_u_r, basis='state_basis').to_numpy())
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(self.reductor.reconstruct(lin_p_r, basis=_basis).to_numpy())

        while not convergence_criterium and i<i_max:            
            outer_loop_start_time = timer()
            self.logger.info(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            self.logger.warning(f"Qr-Vr-IRGNM iteration {i}: J = {J:3.4e} is not sufficent: {np.sqrt(2 * J):3.4e} > {(tol+tau*noise_level):3.4e}.")
            self.logger.info(f'Start Qr-Vr-IRGNM iteration {i}: J = {J:3.4e}, norm_nabla_J = {norm_nabla_J:3.4e}, alpha = {alpha:1.4e}')
            self.logger.info(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            

            # self.FOM.A.material_model.save_time_series(
            #     [v.real_part.impl for v in u.vectors],
            #     str(f'u_snapshot_{self.I}'),
            #     str(self.save_path),
            #     np.linspace(self.FOM.T_initial, self.FOM.T_final, self.FOM.nt+1)
            # )
            # self.FOM.A.material_model.save_time_series(
            #     [v.real_part.impl for v in p.vectors],
            #     str(f'p_snapshot_{self.I}'),
            #     str(self.save_path),
            #     np.linspace(self.FOM.T_initial, self.FOM.T_final, self.FOM.nt+1)
            # )

            # lin_u = self.FOM.solve_linearized_state(q, nabla_J, u, use_cached_operators=use_cached_operators)
            # lin_p = self.FOM.solve_linearized_adjoint(q, u, lin_u, use_cached_operators=use_cached_operators)

            # self.FOM.A.material_model.save_time_series(
            #     [v.real_part.impl for v in lin_u.vectors],
            #     str(f'lin_u_snapshot_{self.I}'),
            #     str(self.save_path),
            #     np.linspace(self.FOM.T_initial, self.FOM.T_final, self.FOM.nt+1)
            # )
            # self.FOM.A.material_model.save_time_series(
            #     [v.real_part.impl for v in lin_p.vectors],
            #     str(f'lin_p_snapshot_{self.I}'),
            #     str(self.save_path),
            #     np.linspace(self.FOM.T_initial, self.FOM.T_final, self.FOM.nt+1)
            # )

    
            assert self.FOM_projector.project_domain(center=q) == q
            q_r = self.reductor.project_vectorarray(q, 'parameter_basis')
            q_r = self.QrVrROM.Q.make_array(q_r)

            u_r, u_dot_r = self.QrVrROM.solve_state(q_r, 
                                                    use_cached_operators=use_cached_operators,
                                                    return_higher_orders = True)
            p_r, p_dot_r = self.QrVrROM.solve_adjoint(q_r, 
                                                      u_r, 
                                                      use_cached_operators=use_cached_operators,
                                                      return_higher_orders = True)
            J_r = self.QrVrROM.objective(u_r)

            #print(f"ROM sparsity = {self.QrVrROM.compute_sparsity(q_r)}")
            
            nabla_J_r = self.QrVrROM.gradient(u_r, p_r, q_r, use_cached_operators=use_cached_operators)
            
            errors = \
            self.estimate_errors(
                model = self.QrVrROM,
                q_r = q_r,
                u_r = u_r,
                p_r = p_r,
                u_dot_r = u_dot_r,
                p_dot_r = p_dot_r,
                J_r = J,
                targets=error_estimate_targets_outer,
                use_cached_operators=use_cached_operators,
                use_error_estimator=use_error_estimator)
            
            if J_r > 0:
                abs_est_error_J_r = errors['err_J']
                rel_est_error_J_r = abs_est_error_J_r / J_r
            else:
                rel_est_error_J_r = np.inf


            if eta <= eta_min:
                self.statistics["stagnation_flag"] = True
                self.logger.info(f"Trust region tolerance eta = {eta} falls below eta_min = {eta_min}.")
                break
                    
            proj_q_in_tr = rel_est_error_J_r <= eta
        
            if AGC_jump_back: 
                self.statistics['flags']['proj_q_in_tr'][-1] = proj_q_in_tr
            else:
                self.statistics['flags']['proj_q_in_tr'].append(proj_q_in_tr)

            if not proj_q_in_tr:
                # self.reductor = InstationaryModelIPReductor(
                #     self.FOM,
                #     self.optimizer_parameter['error_estimator_types'],
                #     use_adjoint_space = self.optimizer_parameter["use_adjoint_space"]
                # )

                self._logger.warning(f"q^(i) is not in the trust region.")
                self._logger.warning(f"Extending reduced spaces with all snapshots.")

                _enrichment = copy.deepcopy(enrichment)
                for basis in self.reduced_bases:
                    _enrichment[basis]['compression']['sample_every_n_th'] = None
                    _enrichment[basis]['compression']['normalize'] = None
                    _enrichment[basis]['compression']['HaPOD'] = None
                    _enrichment[basis]['compression']['keep_last_n'] = None
                                    
                self._reset_snapshots()

                if enrichment['parameter_basis']['reduced_basis']:
                    self.logger.debug(f"Extending Qr-snapshots")
                    self.snapshots['parameter_basis'].append(nabla_J)
                    self.snapshots['parameter_basis'].append(q)
                    self.snapshots['parameter_basis'].append(self.FOM.Q.make_array(self.FOM.setup['q_circ']))
                                
                self.logger.debug(f"Extending Vr-snapshots")
                if self.reductor.use_adjoint_space:
                    self.snapshots['state_basis'].append(u)
                    self.snapshots['adjoint_basis'].append(p)
                else:
                    self.snapshots['state_basis'].append(u)
                    self.snapshots['state_basis'].append(p)
                    
                self.QrVrROM = self.extend_bases_and_rebuild_QrVrROM(
                    bases=self.reduced_bases,
                    enrichment=_enrichment,
                    i = i
                )   

                # print("--------------------------------")                
                # state_basis = self.reductor.bases['state_basis']
                # error_matrix = state_basis.inner(state_basis, self.FOM.products['prod_V'])
                # print(error_matrix)
                 

                q_r = self.reductor.project_vectorarray(q, 'parameter_basis')
                q_r = self.QrVrROM.Q.make_array(q_r)
                u_r, u_dot_r = self.QrVrROM.solve_state(q_r, 
                                                        use_cached_operators=False,
                                                        return_higher_orders=True)
                p_r, p_dot_r = self.QrVrROM.solve_adjoint(q_r, 
                                                          u_r, 
                                                          use_cached_operators=False,
                                                          return_higher_orders=True)
                
                J_r = self.QrVrROM.objective(u_r)

                nabla_J_r = self.QrVrROM.gradient(u_r, p_r, q_r)
                norm_nabla_J_r = self.QrVrROM.compute_gradient_norm(nabla_J_r)

                errors = \
                self.estimate_errors(
                    model = self.QrVrROM,
                    q_r = q_r,
                    u_r = u_r,
                    p_r = p_r,
                    u_dot_r = u_dot_r,
                    p_dot_r = p_dot_r,
                    J_r = J,
                    targets=error_estimate_targets_outer,
                    use_cached_operators=use_cached_operators,
                    use_error_estimator=use_error_estimator)
                
                if J_r > 0:
                    abs_est_error_J_r = errors['err_J']
                    rel_est_error_J_r = abs_est_error_J_r / J_r
                else:
                    rel_est_error_J_r = np.inf
            
            # print(rel_est_error_J_r)
            # print(eta)
            # print(proj_q_in_tr)
            print(rel_est_error_J_r)
            print(eta)
            assert (rel_est_error_J_r - 1e-14) <= eta 

            projector = SimpleBoundDomainProjector(
                model = self.QrVrROM,
                bounds = self.FOM.bounds,
                reductor = self.reductor,
                use_sufficient_condition = True,
                #use_sufficient_condition = False,
                logger = self.logger
            )
            #projector = None

            ########################################### AGC ###########################################

            self.logger.warning("Calculate AGC with Armijo backtracking.")

            AGC_start_time = timer()

            if reg_AGC_step:                
                if last_inner_alpha is not None:
                    AGC_alpha = last_inner_alpha
                else:
                    AGC_alpha = alpha

                previous_J = self.QrVrROM.objective(u_r, q=q_r, alpha=AGC_alpha)
                nabla_reg_J_r = self.QrVrROM.gradient(u_r, p_r, q_r, alpha=AGC_alpha)
                norm_grad = self.QrVrROM.compute_gradient_norm(nabla_reg_J_r)            
                search_direction = -nabla_reg_J_r
                search_direction.scal(1.0 / norm_grad)
            else:
                AGC_alpha = 0.0
                previous_J = J_r
                norm_grad = self.QrVrROM.compute_gradient_norm(nabla_J_r)            
                search_direction = -nabla_J_r
                search_direction.scal(1.0 / norm_grad)

            self.logger.warning(f"Using AGC_alpha = {AGC_alpha}.")

            q_agc, J_r_AGC, model_unsufficent, AGC_max_iter_cond, _, errors = self._armijo_TR_line_serach(
                model = self.QrVrROM,
                previous_q = q_r,
                previous_J = previous_J,
                search_direction = search_direction,
                max_iter = agc_armijo_max_iter,
                inital_step_size = inital_agc_armijo_step_size,
                eta = eta,
                beta = beta_1,
                kappa_arm = kappa_arm,
                use_cached_operators=use_cached_operators,
                projector = projector,
                alpha = AGC_alpha,
                use_error_estimator=use_error_estimator
            )

            print("$$$$$$$$$$$$$$$$$$$$$")
            print(J)
            print(J_r)
            print(J_r_AGC)
            
            AGC_decay_cond = J_r_AGC < (J + 1e-13)
            
            if not AGC_jump_back:
                self.statistics['flags']['AGC_decay_cond'].append(AGC_decay_cond)

            if reg_AGC_step:
                AGC_decay_cond = True
                        
            if not AGC_decay_cond:
                self._logger.warning(f"J_r_AGC = {J_r_AGC:3.4e} is greater or equal than J = {J:3.4e}.")
                self._logger.warning(f"Extending reduced spaces with all snapshots and recomputing AGC.")

                
                _enrichment = copy.deepcopy(enrichment)
                for basis in self.reduced_bases:
                    _enrichment[basis]['compression']['sample_every_n_th'] = None
                    _enrichment[basis]['compression']['normalize'] = None
                    _enrichment[basis]['compression']['HaPOD'] = None
                    _enrichment[basis]['compression']['keep_last_n'] = None
                
                self._reset_snapshots()
                if enrichment['parameter_basis']['reduced_basis']:
                    self.logger.debug(f"Extending Qr-snapshots")
                    self.snapshots['parameter_basis'].append(nabla_J)                
                
                self.logger.debug(f"Extending Vr-snapshots")
                if self.reductor.use_adjoint_space:
                    self.snapshots['state_basis'].append(u)
                    self.snapshots['adjoint_basis'].append(p)
                else:
                    self.snapshots['state_basis'].append(u)
                    self.snapshots['state_basis'].append(p)


                self.QrVrROM = self.extend_bases_and_rebuild_QrVrROM(
                    bases=self.reduced_bases,
                    enrichment=_enrichment,
                    i = i
                )     

                AGC_jump_back = True
                print(eta)
                eta = beta_3 * eta
                print(eta)
                continue
            
            if not reg_AGC_step:
                assert not AGC_max_iter_cond

            AGC_jump_back = False
                
            self.statistics['flags']['model_unsufficent'].append(model_unsufficent)
            self.statistics["outer_loop_runtime"]['AGC_runtime'].append(timer() - AGC_start_time)

            q_r = q_agc.copy()

            ########################################### IRGNM ###########################################
            IRGNM_start_time = timer()

            TR_params = {
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
                                                  TR_enforcement=TR_enforcement,
                                                  TR_params=TR_params,
                                                  lin_solver_parms=lin_solver_parms,
                                                  use_cached_operators=use_cached_operators,
                                                  projector=projector,
                                                  use_error_estimator=use_error_estimator)
            
            
            self.statistics["outer_loop_runtime"]['IRGNM_runtime'].append(timer() - IRGNM_start_time)

            ########################################### Accept / Reject ###########################################

            if len(IRGNM_statistic) > 0:
                check_conditions = len(IRGNM_statistic['q']) > 1
            else:
                check_conditions = False

            self.statistics['flags']['check_conditions'].append(check_conditions)

            if check_conditions:
                self.logger.debug("Decide on q; Either accept or reject")

                u_r, u_dot_r = self.QrVrROM.solve_state(q_r,
                                                        use_cached_operators=use_cached_operators, 
                                                        return_higher_orders=True)
                p_r, p_dot_r = self.QrVrROM.solve_adjoint(q_r, 
                                                          u_r,
                                                          use_cached_operators=use_cached_operators, 
                                                          return_higher_orders=True)
                J_r = self.QrVrROM.objective(u_r)
                nabla_J_r = self.QrVrROM.gradient(u_r, p_r, q_r)
                norm_nabla_J_r = self.QrVrROM.compute_gradient_norm(nabla_J_r)

                errors = \
                self.estimate_errors(
                    model = self.QrVrROM,
                    q_r = q_r,
                    u_r = u_r,
                    p_r = p_r,
                    u_dot_r = u_dot_r,
                    p_dot_r = p_dot_r,
                    J_r = J,
                    targets = error_estimate_targets_outer,
                    use_cached_operators=use_cached_operators,
                    use_error_estimator=use_error_estimator
                )
                
                if J_r > 0:
                    abs_est_error_J_r = errors['err_J']
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
                    q = self.FOM_projector.project_domain(center=q)
                    u = self.FOM.solve_state(q, use_cached_operators=use_cached_operators)
                    p = self.FOM.solve_adjoint(q, u, use_cached_operators=use_cached_operators)
                    J = self.FOM.objective(u)
                    nabla_J, time_step_nabla_J = self.FOM.gradient(u, 
                                                                   p, 
                                                                   q, 
                                                                   use_cached_operators=use_cached_operators,
                                                                   return_per_time_step = True)
                    

                    if enrichment['state_basis']['additional_snapshots']['include_lins'] or enrichment['parameter_basis']['additional_snapshots']['include_lin_grad']:
                        direction = -nabla_J
                        #direction = q - self.statistics["q"][-1]

                        lin_u = self.FOM.solve_linearized_state(q, direction, u, use_cached_operators=use_cached_operators)
                        lin_p = self.FOM.solve_linearized_adjoint(q, u, lin_u, use_cached_operators=use_cached_operators)

                    if enrichment['parameter_basis']['additional_snapshots']['include_lin_grad']:
                        nabla_lin_J = self.FOM.linearized_gradient(q, nabla_J, u, lin_p, alpha=0, use_cached_operators=use_cached_operators)

                    
                    norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)
                    self.statistics['outer_loop_runtime']['solve_snapshot_FOM_runtime'].append(timer()  - solve_snapshot_FOM_start_time)

                    delta_J = self.statistics["J"][-1] - J
                    delta_J_r = self.statistics["J_r"][-1]-J_r

                    if delta_J_r > 0:
                        rho = delta_J / delta_J_r
                    else:
                        rho = np.inf

                    if rho > beta_2:
                        eta = 1/ beta_3 * eta
                        eta = np.min([eta, eta_max])
                        self.logger.info(f"    rho = {rho:3.4e} is greater than beta_2 = {beta_2:3.4e}; updating eta to {eta:3.4e}.")
                    else:
                        self.logger.info(f"    rho = {rho:3.4e} is smaller than beta_2 = {beta_2:3.4e}; keeping eta at {eta:3.4e}.")
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
                    q_ = self.FOM_projector.project_domain(center=q_)
                    u_ = self.FOM.solve_state(q_, use_cached_operators=use_cached_operators)
                    p_ = self.FOM.solve_adjoint(q_, u_, use_cached_operators=use_cached_operators)
                    J_ = self.FOM.objective(u_)
                    nabla_J_, time_step_nabla_J_ = self.FOM.gradient(u, 
                                                                     p, 
                                                                     q, 
                                                                     use_cached_operators=use_cached_operators,
                                                                     return_per_time_step = True)
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
                        time_step_nabla_J = time_step_nabla_J_ 
                        norm_nabla_J = norm_nabla_J_

                        if enrichment['state_basis']['additional_snapshots']['include_lins'] or enrichment['parameter_basis']['additional_snapshots']['include_lin_grad']:
                            direction = -nabla_J
                            #direction = q - self.statistics["q"][-1]

                            lin_u = self.FOM.solve_linearized_state(q, direction, u, use_cached_operators=use_cached_operators)
                            lin_p = self.FOM.solve_linearized_adjoint(q, u, lin_u, use_cached_operators=use_cached_operators)

                        if enrichment['parameter_basis']['additional_snapshots']['include_lin_grad']:
                            nabla_lin_J = self.FOM.linearized_gradient(q, nabla_J, u, lin_p, alpha=0, use_cached_operators=use_cached_operators)
                        
                        delta_J = self.statistics["J"][-1] - J
                        delta_J_r = self.statistics["J_r"][-1] - J_r

                        if delta_J_r > 0:
                            rho = delta_J / delta_J_r
                        else:
                            rho = np.inf

                        if rho > beta_2:
                            eta = 1/ beta_3 * eta
                            eta = np.min([eta, eta_max])
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
                q = self.FOM_projector.project_domain(center=q)
                u = self.FOM.solve_state(q, use_cached_operators=use_cached_operators)
                p = self.FOM.solve_adjoint(q, u, use_cached_operators=use_cached_operators)
                J = self.FOM.objective(u)
                self.statistics['outer_loop_runtime']['solve_snapshot_FOM_runtime'].append(timer()  - solve_snapshot_FOM_start_time)

                nabla_J, time_step_nabla_J = self.FOM.gradient(u, 
                                                               p, 
                                                               q, 
                                                               use_cached_operators=use_cached_operators,
                                                               return_per_time_step = True)
                
                norm_nabla_J = self.FOM.compute_gradient_norm(nabla_J)
                eta = beta_3 * eta

            for basis in self.reduced_bases:
                self.statistics["outer_loop_runtime"]['extend_runtime'][basis].append(0.0)
            self.statistics["outer_loop_runtime"]['reduce_runtime'].append(0.0)

            ########################################### Final ###########################################

            convergence_criterium = np.sqrt(2 * J) < tol+tau*noise_level
            self.statistics['flags']['rejected'].append(rejected)
            
            if not rejected:
                delta = delta
                
                if len(IRGNM_statistic) > 0:
                    try:
                        alpha = IRGNM_statistic["alpha"][1]
                        last_inner_alpha = IRGNM_statistic["alpha"][-1]
                    except IndexError:
                        last_inner_alpha = None

                if not convergence_criterium:
                    self._reset_snapshots()

                    additional_parameter_snapshots, additional_state_snapshots, additional_adjoint_snapshots = \
                    self.snapshot_preprocessor.additional_snapshots(
                        config = enrichment,
                        bases = self.reduced_bases,
                        q = q,
                        u = u,
                        lin_u = lin_u,
                        lin_p = lin_p,
                        nabla_J = nabla_J,
                        nabla_lin_J = nabla_lin_J,
                        time_step_nabla_J = time_step_nabla_J,
                        use_cached_operators = use_cached_operators,
                    )

                    if enrichment['parameter_basis']['reduced_basis']:
                        self.logger.debug(f"Extending Qr-snapshots")            
                        self.snapshots['parameter_basis'].append(nabla_J)
                        self.snapshots['parameter_basis'].append(
                            additional_parameter_snapshots
                        )
                        
                    self.logger.debug(f"Extending Vr-snapshots")

                    if self.reductor.use_adjoint_space:
                        self.snapshots['state_basis'].append(u)
                        self.snapshots['adjoint_basis'].append(p)

                        self.snapshots['adjoint_basis'].append(
                            additional_adjoint_snapshots
                        )
                    else:
                        self.snapshots['state_basis'].append(u)
                        self.snapshots['state_basis'].append(p)


                        # # base parameter (example: all ones; replace with your actual q)
                        # q_base = np.ones((1, 961))

                        # # noise level (standard deviation)
                        # sigma = 1.00  # 5% noise, tune as needed

                        # # draw random _q close to q_base
                        # _q = q_base + np.random.normal(loc=0.0, scale=sigma, size=q_base.shape)
                        # _q = self.FOM.Q.make_array(_q)
                        # _q = self.FOM_projector.project_domain(center=_q)
                        # _u = self.FOM.solve_state(_q, use_cached_operators=False)        
                        # _p = self.FOM.solve_adjoint(_q, _u, use_cached_operators=False)
                        # self.snapshots['state_basis'].append(_u)
                        # self.snapshots['state_basis'].append(_p)

                    self.snapshots['state_basis'].append(
                        additional_state_snapshots
                    )


                    ############################################################

                    self.QrVrROM = self.extend_bases_and_rebuild_QrVrROM(
                        bases=self.reduced_bases,
                        enrichment=enrichment,
                        i = i
                    )

                    ############################################################

                    # basis = 'state_basis'
                    # _basis = self.reductor.bases[basis]

                    # coeff_u = np.sum((u.inner(_basis, self.reductor.products[basis]))**2, axis=0)
                    # err_i_u = np.sum(self.reductor.products[basis].pairwise_apply2(u,u)) - np.cumsum(coeff_u)
                    
                    # coeff_p = np.sum((p.inner(_basis, self.reductor.products[basis]))**2, axis=0)
                    # err_i_p = np.sum(self.reductor.products[basis].pairwise_apply2(p,p)) - np.cumsum(coeff_p)

                    # # err_i_u = err_i_u[err_i_u > 0]
                    # # err_i_p = err_i_p[err_i_p > 0]
                    
                    # print(err_i_u[-1])
                    # print(err_i_p[-1])

                    # color = cmap(i+1)
                    # ax_2.semilogy(err_i_u, color=color)
                    # ax_2.semilogy(err_i_p, color=color, linestyle="--")
                    # ax_2.set_ylim([1e-18, 1e3])
                    # ax_2.grid(True)
                    
                    # fig_2.savefig(self.save_path / "coeffs_after_enrich.pdf")



                    # _basis = self.reductor.bases['state_basis']
                    # self.FOM.A.material_model.save_time_series(
                    #     [v.real_part.impl for v in _basis.vectors],
                    #     str('state_basis'),
                    #     str(self.save_path),
                    #     np.linspace(0, len(_basis), len(_basis))
                    # )

                    ############################################################

                    q_r = self.reductor.project_vectorarray(q, 'parameter_basis')
                    q_r = self.QrVrROM.Q.make_array(q_r)

                self.statistics["q"].append(q)
                self.statistics["eta"].append(eta)
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

                if 'counts' in IRGNM_statistic.keys():
                    self.statistics["counts"].append(IRGNM_statistic['counts'])
                else:
                    self.statistics["counts"].append({})

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

