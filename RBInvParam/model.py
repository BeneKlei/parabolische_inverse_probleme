import logging
from typing import Dict, Union
import numpy as np
import itertools

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.interface import VectorArray
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorSpace
from pymor.core.base import ImmutableObject
from pymor.operators.constructions import ZeroOperator

from RBInvParam.evaluators import UnAssembledA, UnAssembledB, AssembledA, AssembledB
from RBInvParam.timestepping import ImplicitEulerTimeStepper
from RBInvParam.error_estimator import StateErrorEstimator, AdjointErrorEstimator, \
    ObjectiveErrorEstimator, CoercivityConstantEstimator
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.products import BochnerProductOperator
#from RBInvParam.reductor import InstationaryModelIPReductor

class InstationaryModelIP(ImmutableObject):
    id_iter = itertools.count()

    def __init__(self,
                 u_0 : VectorArray, 
                 M : Operator,
                 A : Union[UnAssembledA, AssembledA],
                 L : VectorArray,
                 B : Union[UnAssembledB, AssembledB],
                 constant_cost_term: Union[None, float],
                 linear_cost_term: Union[None, NumpyMatrixOperator],
                 bilinear_cost_term: Union[None, NumpyMatrixOperator],
                 Q : VectorSpace,
                 V : VectorSpace,
                 q_circ: VectorArray,
                 constant_reg_term: float,
                 linear_reg_term: NumpyMatrixOperator,
                 bilinear_reg_term: NumpyMatrixOperator,
                 state_error_estimator: Union[None, StateErrorEstimator],
                 adjoint_error_estimator: Union[None, AdjointErrorEstimator],
                 objective_error_estimator: Union[None, ObjectiveErrorEstimator],
                 products : Dict,
                 visualizer,
                 setup : Dict,
                 model_constants : Dict,
                 name: str = None,
                 num_calls: Dict = None,
                 logger: logging.Logger = None,
                 bounds: np.ndarray = None):
        
        # TODO On palma STDERR does not go into *_IRGNM.log. Why? 
        logging.basicConfig()
        if logger:
            self._logger = logger
        else:
            self._logger = get_default_logger(
                logger_name=self.__class__.__name__ + str(next(InstationaryModelIP.id_iter))
            )
            self._logger.setLevel(logging.DEBUG)
        self.logger.debug(f"Setting up {self.__class__.__name__}")
        
        self.u_0 = u_0
        assert np.all(u_0.to_numpy() == 0)
        self.p_0 = u_0.copy()
        self.linearized_u_0 = u_0.copy()
        self.linearized_p_0 = u_0.copy()

        assert not M.parametric
        self.M = M.assemble() 

        self.A = A 
        self.L = L 
        self.B = B 
        self.constant_cost_term = constant_cost_term 
        self.linear_cost_term = linear_cost_term 
        self.bilinear_cost_term = bilinear_cost_term 
        self.V = V
        self.Q = Q
        self.q_circ = q_circ 
        self.constant_reg_term = constant_reg_term 
        self.linear_reg_term = linear_reg_term 
        self.bilinear_reg_term = bilinear_reg_term 
        self.state_error_estimator = state_error_estimator
        self.adjoint_error_estimator = adjoint_error_estimator
        self.objective_error_estimator = objective_error_estimator
        self.products = products
        self.visualizer = visualizer
        self.model_constants = model_constants
        self.setup = setup
        self.bounds = bounds
        

        self.nt = self.setup['dims']['nt']
        self.T_initial = self.setup['model_parameter']['T_initial']
        self.T_final = self.setup['model_parameter']['T_final']

        self.delta_t = self.setup['model_parameter']['delta_t']
        self.q_time_dep = self.setup['model_parameter']['q_time_dep']
        self.riesz_rep_grad = self.setup['model_parameter']['riesz_rep_grad']

        self.timestepper = ImplicitEulerTimeStepper(
            nt = self.nt,
            M = self.M,
            A = self.A,
            Q = self.Q,
            V = self.V,
            T_initial= self.T_initial,
            T_final= self.T_final,
            setup = self.setup,
        )

        self.solver_options = None
        self._cached_operators = {
            'q' : self.Q.empty(),
            'A_q' : [],
            'M_dt_A_q' : [],
            'residual_A_q' : [],
            'B_u' : [],
        }
        
        if not num_calls:
            self.num_calls = {
                'solve_state' : 0,
                'solve_adjoint' : 0,
                'solve_linearized_state' : 0,
                'solve_linearized_adjoint' : 0,
                'objective' : 0,
                'gradient' : 0,
                'linearized_objective' : 0,
                'linearized_gradient' : 0,
            }
        else:
            self.num_calls = num_calls

        if name:
            self.setup['model_parameter']['name'] = name

        assert self.M.source == self.M.range
        assert self.A.source == self.A.range
        assert self.M.source == self.A.source
        assert self.A.range == self.V
        assert isinstance(self.L, VectorArray)
        assert len(self.L) in [1, self.nt]
        assert self.L in self.V
        assert self.A.Q == self.Q
        assert self.q_circ in self.Q

        if self.bilinear_cost_term:
            assert self.bilinear_cost_term.source == self.bilinear_cost_term.range
            assert self.bilinear_cost_term.source == self.A.range
        if self.linear_cost_term:
            assert self.linear_cost_term.range == self.A.range
            assert len(self.linear_cost_term.as_range_array()) == self.nt

        assert self.bilinear_reg_term.source == self.bilinear_reg_term.range
        assert self.bilinear_reg_term.source == self.Q
        assert self.linear_reg_term.range == self.Q    

        if self.state_error_estimator:
            assert isinstance(state_error_estimator, StateErrorEstimator)
            assert 'A_coercivity_constant_estimator' in self.model_constants.keys()
            #assert self.state_error_estimator.state_residual_operator.source == self.A.source
        if self.adjoint_error_estimator:
            assert isinstance(adjoint_error_estimator, AdjointErrorEstimator)
            assert 'A_coercivity_constant_estimator' in self.model_constants.keys()
            #assert self.adjoint_error_estimator.adjoint_residual_operator.source == self.A.source
        if self.objective_error_estimator:
            assert isinstance(objective_error_estimator, ObjectiveErrorEstimator)
            assert 'C_continuity_constant' in self.model_constants.keys()
            assert self.state_error_estimator
            assert self.adjoint_error_estimator
        if self.model_constants:
            assert 'A_coercivity_constant_estimator' in self.model_constants.keys()
            assert 'C_continuity_constant' in self.model_constants.keys()
            assert isinstance(self.model_constants['A_coercivity_constant_estimator'], CoercivityConstantEstimator)
            #assert self.model_constants['A_coercivity_constant_estimator'].Q == self.Q

        assert 'bochner_prod_Q' in self.products
        assert isinstance(self.products['bochner_prod_Q'], BochnerProductOperator)
        assert self.products['bochner_prod_Q'].product.source == self.Q
        assert 'prod_Q' in self.products
        assert self.products['prod_Q'].source == self.Q

        if self.bounds is not None:
            assert isinstance(self.bounds, np.ndarray)
            if self.q_time_dep:
                assert self.bounds.shape == (self.nt * self.Q.dim , 2)
            else:
                assert self.bounds.shape == (self.Q.dim , 2)
            assert np.all(self.bounds[:,0] < self.bounds[:,1])
            

#%% cache methods
    def _cache_update_required(self,
                               q : VectorArray) -> bool:
        
        if len(self._cached_operators['q']) == 0:
            return True
        else:
            return np.any((self._cached_operators['q']-q).norm() != 0)
    

    def _cache_time_independed_operators(self, 
                                       q: VectorArray,
                                       u: VectorArray,
                                       target: str = 'M_dt_A_q') -> None:
        
        if target == 'A_q':
            A_q = self.A(q[0])
            self._cached_operators[target].append(
                A_q.assemble()
            )
        elif target == 'M_dt_A_q':
            dt = (self.T_final - self.T_initial) / self.nt
            A_q = self.A(q[0])
            M_dt_A_q = (self.M + A_q * dt).with_(solver_options=self.solver_options)
            self._cached_operators[target].append(
                M_dt_A_q.assemble()        
            )
        elif target == 'residual_A_q':
            if self.state_error_estimator:
                assert self.state_error_estimator.state_residual_operator.A == self.adjoint_error_estimator.adjoint_residual_operator.A
                self._cached_operators[target].append(
                    self.state_error_estimator.state_residual_operator._precompute_residual_A_q(q[0])
                )
        elif target == 'B_u':
            self._cached_operators[target] = [self.B(u[idx]) for idx in range(len(u))]
        else:
            self.logger.error(f'Target {target} is not known.')
            raise ValueError
    
    def _cache_time_depended_operators(self, 
                                       q: VectorArray,
                                       u: VectorArray,
                                       target: str = 'M_dt_A_q') -> None:

        
        for n in range(self.nt):            
            if target == 'A_q':
                A_q = self.A(q[n])
                self._cached_operators[target].append(
                    A_q.assemble()
                )
            elif target == 'M_dt_A_q':
                dt = (self.T_final - self.T_initial) / self.nt
                A_q = self.A(q[n])
                M_dt_A_q = (self.M + A_q * dt).with_(solver_options=self.solver_options)
                self._cached_operators[target].append(
                        M_dt_A_q.assemble()
                )
            elif target == 'residual_A_q':
                if self.state_error_estimator:
                    assert self.state_error_estimator.state_residual_operator.A == self.adjoint_error_estimator.adjoint_residual_operator.A
                    self._cached_operators[target].append(
                        self.state_error_estimator.state_residual_operator._precompute_residual_A_q(q[n])
                    ) 
            elif target == 'B_u':
                self._cached_operators[target].append(self.B(u[n]))
            else:
                self.logger.error(f'Target {target} is not known.')
                raise ValueError
    
    
    def cache_operators(self, 
                        q: VectorArray,
                        u: VectorArray = None,
                        target: str = 'M_dt_A_q') -> None:
        
        assert target in self._cached_operators.keys()
        assert q in self.Q
        if len(self._cached_operators['q']) != 0:
            assert np.all((self._cached_operators['q']-q).norm() == 0)
        
        if target == 'B_u':
            assert u
            assert len(u) == self.nt

        self.logger.debug(f'Caching {target}')
        self._cached_operators['q'] = q.copy()
        

        if self.setup['model_parameter']['q_time_dep']:            
            self._cache_time_depended_operators(
                q = q,
                u = u,
                target = target
            )
        else:
            self._cache_time_independed_operators(
                q = q,
                u = u,
                target = target
            )
            

    def delete_cached_operators(self) -> None:
        self.logger.debug('Deleting cache')

        del self._cached_operators['q'][:]
        del self._cached_operators['A_q'][:]
        del self._cached_operators['M_dt_A_q'][:]
        del self._cached_operators['residual_A_q'][:]
        del self._cached_operators['B_u'][:]
        

        self._cached_operators = {
            'q' : self.Q.empty(),
            'A_q' : [],
            'M_dt_A_q' : [],
            'residual_A_q' : [],
            'B_u' : []
        }


#%% solve methods
    def solve_state(self, 
                    q: VectorArray,
                    use_cached_operators: bool = False) -> VectorArray:
        
        assert q in self.Q

        if self.q_time_dep:
            assert len(q) == self.nt
        else:
            assert len(q) == 1

        self.num_calls['solve_state'] += 1

        if use_cached_operators:
            if self._cache_update_required(q):
                self.delete_cached_operators()
           
            if len(self._cached_operators['M_dt_A_q']) == 0:
                self.cache_operators(q=q, target='M_dt_A_q')
            
        iterator = self.timestepper.iterate(initial_data = self.u_0, 
                                            q=q,
                                            rhs=self.L,
                                            use_cached_operators=use_cached_operators,
                                            cached_operators=self._cached_operators)
        
        u = self.V.empty(reserve= self.nt)
        for u_n, _ in iterator:
            u.append(u_n)

        return u

    def solve_adjoint(self, 
                      q: VectorArray, 
                      u: VectorArray,
                      use_cached_operators: bool = False) -> VectorArray:
        
        assert self.bilinear_cost_term 
        assert self.linear_cost_term 
        assert q in self.Q
        assert u in self.V

        if self.q_time_dep:
            assert len(q) == self.nt
        else:
            assert len(q) == 1

        assert len(u) == self.nt

        self.num_calls['solve_adjoint'] += 1

        if use_cached_operators:
            # TODO Move these into one abstract function
            if self._cache_update_required(q):
                self.delete_cached_operators()
           
            if len(self._cached_operators['M_dt_A_q']) == 0:
                self.cache_operators(q=q, target='M_dt_A_q') 

        rhs = self.bilinear_cost_term.apply(u) - self.linear_cost_term.as_range_array()
        rhs = np.flip(rhs.to_numpy(), axis=0)
        if isinstance(self.A, UnAssembledA):
            I = self.A.boundary_info.dirichlet_boundaries(2)
            rhs[:,I] = 0
        #rhs = self.delta_t * self.V.make_array(rhs)
        rhs = self.V.make_array(rhs)

        iterator = self.timestepper.iterate(initial_data = self.p_0, 
                                            q=q,
                                            rhs=rhs,
                                            use_cached_operators=use_cached_operators,
                                            cached_operators=self._cached_operators)
        
        p = self.V.empty(reserve= self.nt)
        for p_n, _ in iterator:
            p.append(p_n)
        return self.V.make_array(np.flip(p.to_numpy(), axis=0))
    
    def solve_linearized_state(self,
                               q: VectorArray,
                               d: VectorArray,
                               u: VectorArray,
                               use_cached_operators: bool = False) -> VectorArray:
        
        assert q in self.Q
        assert d in self.Q
        assert u in self.V
        if self.q_time_dep:
            assert len(q) == self.nt
        else:
            assert len(q) == 1
        assert len(d) == len(q)
        assert len(u) == self.nt

        self.num_calls['solve_linearized_state'] += 1

        if use_cached_operators:
            if self._cache_update_required(q):
                self.delete_cached_operators()
           
            if len(self._cached_operators['M_dt_A_q']) == 0:
                self.cache_operators(q=q, target='M_dt_A_q')

            if len(self._cached_operators['B_u']) == 0:
                self.cache_operators(q=q, u=u, target='B_u')
        
        if use_cached_operators:
            B_u = self._cached_operators['B_u']
        else:
            B_u = [self.B(u[idx]) for idx in range(len(u))]
            
        # TODO Check if this is efficent and / or how its efficeny can be improved
        if self.q_time_dep:
            rhs = self.V.make_array(np.array([
                B_u[idx].B_u(d[idx]).to_numpy()[0] for idx in range(len(u))
            ]))
        else:            
            rhs = self.V.make_array(np.array([
                B_u[idx].B_u(d[0]).to_numpy()[0] for idx in range(len(u))
            ]))
        
        iterator = self.timestepper.iterate(initial_data = self.linearized_u_0, 
                                            q=q,
                                            rhs=rhs,
                                            use_cached_operators=use_cached_operators,
                                            cached_operators=self._cached_operators)
        
        lin_u = self.V.empty(reserve= self.nt)
        for lin_u_n, _ in iterator:
            lin_u.append(lin_u_n)
        return lin_u
    
    def solve_linearized_adjoint(self,
                                 q: VectorArray,
                                 u: VectorArray,
                                 lin_u: VectorArray,
                                 use_cached_operators: bool = False) -> VectorArray:

        assert self.bilinear_cost_term 
        assert self.linear_cost_term
        assert q in self.Q
        assert u in self.V
        assert lin_u in self.V
        if self.q_time_dep:
            assert len(q) == self.nt
        else:
            assert len(q) == 1
        assert len(u) == self.nt
        assert len(lin_u) == self.nt

        self.num_calls['solve_linearized_adjoint'] += 1

        if use_cached_operators:
            if self._cache_update_required(q):
                self.delete_cached_operators()
           
            if len(self._cached_operators['M_dt_A_q']) == 0:
                self.cache_operators(q=q, target='M_dt_A_q')

        rhs = self.bilinear_cost_term.apply(u + lin_u) - self.linear_cost_term.as_range_array()
        rhs = np.flip(rhs.to_numpy(), axis=0)
        if isinstance(self.A, UnAssembledA):
            I = self.A.boundary_info.dirichlet_boundaries(2)
            rhs[:,I] = 0

        #rhs = self.delta_t * self.V.make_array(rhs)
        rhs = self.V.make_array(rhs)
        iterator = self.timestepper.iterate(initial_data = self.p_0, 
                                            q=q,
                                            rhs=rhs,
                                            use_cached_operators=use_cached_operators,
                                            cached_operators=self._cached_operators)
        
        lin_p = self.V.empty(reserve= self.nt)
        for lin_p_n, _ in iterator:
            lin_p.append(lin_p_n)

        return self.V.make_array(np.flip(lin_p.to_numpy(), axis=0))
    
#%% objective and gradient
    def objective(self, 
                  u: Union[VectorArray, np.ndarray],
                  q: VectorArray = None,
                  alpha: float = 0) -> float:
        
        if q:
            assert q in self.Q
            if self.q_time_dep:
                assert len(q) == self.nt
            else:
                assert len(q) == 1

        assert len(u) == self.nt
        assert u in self.V
        assert self.bilinear_cost_term 
        assert self.linear_cost_term
        
        self.num_calls['objective'] += 1
        # compute tracking term
        out = 0.5 * self.delta_t * np.sum(self.bilinear_cost_term.pairwise_apply2(u,u)
                                          + (-2) * self.linear_cost_term.as_range_array().pairwise_inner(u) 
                                          + self.constant_cost_term)
        if alpha > 0:
            assert q is not None
            # add regularization term if alpha >0
            return out + alpha * self.regularization_term(q)
        else:
            return out

    def gradient(self,
                 u: VectorArray,
                 p: VectorArray,
                 q: VectorArray = None,
                 alpha: float = 0,
                 use_cached_operators: bool = False) -> VectorArray:
        
        assert u in self.V
        assert p in self.V
        assert len(u) == self.nt
        assert len(p) == self.nt

        if use_cached_operators:
            if self._cache_update_required(q):
                self.delete_cached_operators()
           
            if len(self._cached_operators['B_u']) == 0:                
                self.cache_operators(q=q, u=u, target='B_u')

        if use_cached_operators:
            B_u = self._cached_operators['B_u']
        else:
            B_u = [self.B(u[idx]) for idx in range(len(u))]

        self.num_calls['gradient'] += 1
        grad = self.Q.empty(reserve=self.nt)

        # TODO Check if this is efficent and / or how its efficeny can be improved
        for idx in range(0, self.nt):
            grad.append(B_u[idx].B_u_ad(p[idx]))

        if not self.q_time_dep:
            grad = self.delta_t * self.Q.make_array(np.sum(grad.to_numpy(), axis=0, keepdims=True))
        
        if self.riesz_rep_grad:
            grad = self.products['prod_Q'].apply_inverse(grad) 
        
        if alpha > 0:
            out = grad + alpha * self.gradient_regularization_term(q)
        else:
            out = grad

        return out

    def linearized_objective(self,
                            q: VectorArray,
                            d: VectorArray,
                            u: VectorArray,
                            lin_u: VectorArray,
                            alpha : float,
                            use_cached_operators: bool = False) -> float:

        if self.q_time_dep:
            assert len(q) == self.nt
        else:
            assert len(q) == 1
        assert len(d) == len(q)
        assert len(u) == self.nt
        assert len(lin_u) == self.nt

        assert q in self.Q
        assert d in self.Q
        assert u in self.V
        assert lin_u in self.V
        assert self.bilinear_cost_term 
        assert self.linear_cost_term

        self.num_calls['linearized_objective'] += 1

        u_q_d = u + lin_u
        out = 0.5 * self.delta_t * np.sum( \
                      self.bilinear_cost_term.pairwise_apply2(u_q_d,u_q_d) + \
                      (-2)  * self.linear_cost_term.as_range_array().pairwise_inner(u_q_d) + \
                      self.constant_cost_term)
        if alpha > 0:
            return out + alpha * self.linearized_regularization_term(q, d)
            
        else:
            return out
        
    def linearized_gradient(self,
                            q: VectorArray,
                            d: VectorArray,
                            u: VectorArray,
                            lin_p: VectorArray,
                            alpha : float,
                            use_cached_operators: bool = False) -> VectorArray:
        
        if self.q_time_dep:
            assert len(q) == self.nt
        else:
            assert len(q) == 1
        assert len(q) == len(d)
        assert len(u) == self.nt
        assert len(lin_p) == self.nt
        

        assert q in self.Q
        assert d in self.Q
        assert u in self.V
        assert lin_p in self.V

        if use_cached_operators:
            if self._cache_update_required(q):
                self.delete_cached_operators()
           
            if len(self._cached_operators['B_u']) == 0:                
                self.cache_operators(q=q, u=u, target='B_u')

        if use_cached_operators:
            B_u = self._cached_operators['B_u']
        else:
            B_u = [self.B(u[idx]) for idx in range(len(u))]

        self.num_calls['linearized_gradient'] += 1
        #grad = np.empty((self.nt, self.setup['dims']['par_dim']))        
        grad = self.Q.empty(reserve=self.nt)

        # TODO Check if this is efficent and / or how its efficeny can be improved
        for idx in range(0, self.nt):
            grad.append(B_u[idx].B_u_ad(lin_p[idx]))

        if not self.q_time_dep:
            grad = self.delta_t * self.Q.make_array(np.sum(grad.to_numpy(), axis=0, keepdims=True))

        if self.riesz_rep_grad:
            grad = self.products['prod_Q'].apply_inverse(grad) 

        if alpha > 0:
            out = grad + alpha * self.linarized_gradient_regularization_term(q,d)
        else:
            out = grad

        return out
    
    def linearized_hessian(self):
        raise NotImplementedError

#%% regularization
    def regularization_term(self, 
                            q: VectorArray) -> float:
        assert q in self.Q
        if self.q_time_dep:
            assert len(q) == self.nt
        else:
            assert len(q) == 1
        
        if self.q_time_dep:
            return 0.5 * self.delta_t * np.sum(self.bilinear_reg_term.pairwise_apply2(q,q) 
                                            + (-2) * self.linear_reg_term.as_range_array().pairwise_inner(q) 
                                            + self.constant_reg_term)
        else:
            return 0.5 * (self.bilinear_reg_term.pairwise_apply2(q,q)
                       + (-2) * q.inner(self.linear_reg_term.as_range_array())
                       + self.constant_reg_term)[0,0]
                 
    def gradient_regularization_term(self, 
                                     q: VectorArray) -> float:
        assert q in self.Q
        if self.q_time_dep:
            assert len(q) == self.nt
        else:
            assert len(q) == 1

        out = (- self.linear_reg_term.as_range_array() + self.products['prod_Q'].apply(q))

        if not self.riesz_rep_grad:
            return out
        else:
            return self.products['prod_Q'].apply_inverse(out)
                 
    def linearized_regularization_term(self, 
                                       q: VectorArray,
                                       d: VectorArray) -> float:
        if self.q_time_dep:
            assert len(q) == self.nt
        else:
            assert len(q) == 1
        assert len(q) == len(d)
        
        assert q in self.Q
        assert d in self.Q
        
        if self.q_time_dep:
            return 0.5 * self.delta_t * np.sum(self.bilinear_reg_term.pairwise_apply2(q+d,q+d)
                                            + (-2) * self.linear_reg_term.as_range_array().pairwise_inner(q+d) 
                                            + self.constant_reg_term)
        else:
            return 0.5 * (self.bilinear_reg_term.pairwise_apply2(q+d,q+d)
                       + (-2) * (q+d).inner(self.linear_reg_term.as_range_array())
                       + self.constant_reg_term)[0,0]
                 
    def linarized_gradient_regularization_term(self,
                                               q: VectorArray,
                                               d: VectorArray) -> float:
        if self.q_time_dep:
            assert len(q) == self.nt
        else:
            assert len(q) == 1
        assert len(q) == len(d)
        
        assert q in self.Q
        assert d in self.Q

        out = (- self.linear_reg_term.as_range_array() + self.products['prod_Q'].apply(q + d))

        if not self.riesz_rep_grad:
            return out
        else:
            return self.products['prod_Q'].apply_inverse(out)
            

#%% error estimator

    # TODO Enable cache also for error est, i.e. A_q
    def estimate_state_error(self,
                             q: VectorArray,
                             u: VectorArray,
                             use_cached_operators: bool = False) -> float:        
        
        assert len(u) == self.nt
        assert q in self.Q
        assert u in self.V

        if use_cached_operators:
            if self._cache_update_required(q):
                self.delete_cached_operators()
           
            if len(self._cached_operators['residual_A_q']) == 0:
                self.cache_operators(q=q, target='residual_A_q')

        if self.state_error_estimator:
            return self.state_error_estimator.estimate_error(
                q = q,
                u = u,
                use_cached_operators=use_cached_operators,
                cached_operators=self._cached_operators
            )
        else:
            return 0.0
            
    def estimate_adjoint_error(self,
                               q: VectorArray,
                               u: VectorArray,
                               p: VectorArray,
                               use_cached_operators: bool = False) -> float:
        
        assert len(u) == self.nt
        assert len(u) == len(p)

        assert q in self.Q
        assert u in self.V
        assert p in self.V

        if use_cached_operators:
            if self._cache_update_required(q):
                self.delete_cached_operators()
           
            if len(self._cached_operators['residual_A_q']) == 0:
                self.cache_operators(q=q, target='residual_A_q')

        if self.state_error_estimator:
            return self.adjoint_error_estimator.estimate_error(
                q = q,
                u = u,
                p = p,
                use_cached_operators=use_cached_operators,
                cached_operators=self._cached_operators
            )
        else:
            return 0.0
            
    def estimate_objective_error(self,
                                 q: VectorArray,
                                 u: VectorArray,
                                 p: VectorArray,
                                 use_cached_operators: bool = False) -> float:

        if use_cached_operators:
            if self._cache_update_required(q):
                self.delete_cached_operators()
           
            if len(self._cached_operators['residual_A_q']) == 0:
                self.cache_operators(q=q, target='residual_A_q')

        if self.objective_error_estimator:
            estimated_state_error = self.estimate_state_error(
                q = q,
                u = u,
                use_cached_operators=use_cached_operators
            )            
            adjoint_residuum = self.adjoint_error_estimator.compute_residuum(
                q = q,
                u = u,
                p = p,
                use_cached_operators=use_cached_operators,
                cached_operators=self._cached_operators
            )

            adjoint_residuum = np.sqrt(self.adjoint_error_estimator.delta_t * \
                np.sum(adjoint_residuum.norm2(
                    product=self.adjoint_error_estimator.product
                )
            ))
            e = self.objective_error_estimator.estimate_error(
                q = q,
                estimated_state_error = estimated_state_error,
                adjoint_residuum = adjoint_residuum
            )
            return e
        else:
            return 0.0
    
    def estimate_gradient_error(self) -> float:
        raise NotImplementedError

    def estimate_linarized_state_error(self) -> float:
        raise NotImplementedError

    def estimate_linarized_adjoint_error(self) -> float:
        raise NotImplementedError

    def estimate_linarized_objective_error(self) -> float:
        raise NotImplementedError

    def estimate_linarized_gradient_error(self) -> float:
        raise NotImplementedError
    
#%% compute functions                 
    def compute_objective(self, 
                          q: VectorArray,
                          alpha : float = 0,
                          use_cached_operators: bool = False) -> float:
        u = self.solve_state(q=q, 
                             use_cached_operators=use_cached_operators)
        return self.objective(u, q, alpha)
    
    def compute_gradient(self,
                         q: VectorArray,
                         alpha : float = 0,
                         use_cached_operators: bool = False) -> float:    
        u = self.solve_state(q=q, 
                             use_cached_operators=use_cached_operators)
        p = self.solve_adjoint(q=q, 
                               u=u, 
                               use_cached_operators=use_cached_operators)
        
        return self.gradient(u, p, alpha)
    
    def compute_linearized_objective(self,
                                     q: VectorArray,
                                     d: VectorArray,
                                     alpha : float,
                                     use_cached_operators: bool = False) -> float:
        u = self.solve_state(q=q, 
                             use_cached_operators=use_cached_operators)
        
        lin_u = self.solve_linearized_state(q=q, 
                                            d=d, 
                                            u=u, 
                                            use_cached_operators=use_cached_operators)
        
        return self.linearized_objective(q, d, u, lin_u, alpha)

    def compute_linearized_gradient(self,
                                    q: VectorArray,
                                    d: VectorArray,
                                    alpha : float,
                                    use_cached_operators: bool = False) -> float:
        u = self.solve_state(q, 
                             use_cached_operators=use_cached_operators)
        lin_u = self.solve_linearized_state(q=q, 
                                            d=d, 
                                            u=u, 
                                            use_cached_operators=use_cached_operators)
        lin_p = self.solve_linearized_adjoint(q=q, 
                                              u=u, 
                                              lin_u=lin_u, 
                                              use_cached_operators=use_cached_operators)

        return self.linearized_gradient(q=q, 
                                        d=d, 
                                        u=u, 
                                        lin_p=lin_p, 
                                        alpha=alpha,
                                        use_cached_operators=use_cached_operators)

    def compute_objective_error_estimate(self,
                                         q: VectorArray,
                                         use_cached_operators: bool = False) -> float:
        u = self.solve_state(q, 
                             use_cached_operators=use_cached_operators)
        p = self.solve_adjoint(q=q, 
                               u=u, 
                               use_cached_operators=use_cached_operators)   

        return self.estimate_objective_error(q=q,
                                             u = u,
                                             p = p,
                                             use_cached_operators=use_cached_operators)

#%% helpers
    def compute_gradient_norm(self,
                              V: VectorArray) -> float:
        assert V in self.Q
        if self.q_time_dep:
            return np.sqrt(self.products['bochner_prod_Q'].apply2(V, V))[0,0]
        else:
            return np.sqrt(self.products['prod_Q'].apply2(V, V))[0,0]
    
    def pymor_to_numpy(self,q):
        return q.to_numpy()
    
    def numpy_to_pymor(self,q):
        return self.Q.make_array(q)
