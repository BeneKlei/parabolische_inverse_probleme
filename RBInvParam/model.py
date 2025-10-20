import logging
from typing import Dict, Union, List, Tuple
import numpy as np
import itertools

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.interface import VectorArray, VectorSpace
from pymor.vectorarrays.numpy import NumpyVectorArray, VectorSpace
from pymor.operators.interface import Operator
from pymor.core.base import ImmutableObject

from RBInvParam.evaluators import EvaluatorA, EvaluatorB
from RBInvParam.timestepping import get_time_stepper

from RBInvParam.error_estimators.state_error_estimators import StateErrorEstimator
from RBInvParam.error_estimators.adjoint_error_estimators import AdjointErrorEstimator
from RBInvParam.error_estimators.objective_error_estimators import ObjectiveErrorEstimator, CoercivityConstantEstimator

from RBInvParam.utils.logger import get_default_logger
from RBInvParam.products import BochnerProductOperator
#from RBInvParam.reductor import InstationaryModelIPReductor

class InstationaryModelIP(ImmutableObject):
    id_iter = itertools.count()

    def __init__(self,
                 initial_data : dict, 
                 M : Operator,
                 A : EvaluatorA,
                 L : VectorArray,
                 B : EvaluatorB,
                 constant_cost_term: None | float,
                 linear_cost_term: None | VectorArray,
                 bilinear_cost_term: None | Operator,
                 Q : VectorSpace,
                 V : VectorSpace,
                 q_circ: VectorArray,
                 constant_reg_term: float,
                 linear_reg_term: NumpyMatrixOperator,
                 bilinear_reg_term: NumpyMatrixOperator,
                 state_error_estimator: None | StateErrorEstimator,
                 adjoint_error_estimator: None | AdjointErrorEstimator,
                 objective_error_estimator: None | ObjectiveErrorEstimator,
                 products : Dict,
                 setup : Dict,
                 model_constants : Dict,
                 name: str = None,
                 num_calls: Dict = None,
                 logger: logging.Logger = None,
                 visualizer = None,
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
        
        assert 'state' in initial_data.keys()
        assert 'adjoint' in initial_data.keys()
        assert 'lin_state' in initial_data.keys()
        assert 'lin_adjoint' in initial_data.keys()
        self.initial_data = initial_data

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
        self.T_initial = self.setup['T_initial']
        self.T_final = self.setup['T_final']

        self.delta_t = self.setup['delta_t']
        self.q_time_dep = self.setup['q_time_dep']
        self.riesz_rep_grad = self.setup['riesz_rep_grad']

        assert self.setup['time_stepper']['name'] in ['implicit_euler', 'newman_second_order']

        self.time_stepper = get_time_stepper(
            nt = self.nt,
            M = self.M,
            A = self.A,
            Q = self.Q,
            V = self.V,
            T_initial= self.T_initial,
            T_final= self.T_final,
            q_time_dep=self.q_time_dep,
            time_stepper = self.setup['time_stepper']
        )

        self.solver_options = None
        self._cached_operators = {}

        keys = []
        keys += ['q', 'A_q']
        keys += self.time_stepper.required_cache_keys
        keys += ['residual_A_q', 'B_u']

        self.reset_cached_operators(
            keys = keys
        )
        
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
        assert len(self.L) in [1, self.nt + 1]
        assert self.L in self.V
        assert self.A.Q == self.Q
        assert self.q_circ in self.Q

        if self.bilinear_cost_term:
            assert self.bilinear_cost_term.source == self.bilinear_cost_term.range
            assert self.bilinear_cost_term.source == self.A.range
        if self.linear_cost_term:
            assert self.linear_cost_term in self.A.range
            assert len(self.linear_cost_term) == (self.nt + 1)

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
        
        assert all(x is None for x in self._cached_operators['q']) or all(x is not None for x in self._cached_operators['q'])
        if all(x is None for x in self._cached_operators['q']):
            return True
        else:
            return np.any((self._cached_operators['q']-q).norm() != 0)
    
    def _cache_operators(self, 
                         target: str,
                         time_step: int,
                         q: VectorArray,
                         u: VectorArray) -> None:
        
        if target in self.time_stepper.required_cache_keys:
            assert self._cached_operators['A_q'][time_step]

            self._cached_operators[target][time_step] = \
                self.time_stepper.cache_operator(
                    target = target,
                    time_step = time_step,
                    q = q,
                    u = u,
                    A_q = self._cached_operators['A_q'][time_step]
                )
        elif target == 'A_q':
            self._cached_operators['A_q'][time_step] = self.A(q[time_step])
        elif target == 'residual_A_q':
            if self.state_error_estimator:
                assert self.state_error_estimator.state_residual_operator.A == self.adjoint_error_estimator.adjoint_residual_operator.A
                self._cached_operators['residual_A_q'][time_step] = \
                    self.state_error_estimator.state_residual_operator._precompute_residual_A_q(q[time_step])
        elif target == 'B_u':
            self._cached_operators['B_u'][time_step] = self.B(u[time_step], time_step)
        else:
            self.logger.error(f'Target {target} is not known.')
            raise ValueError
    
    def _cache_time_depended_operators(self, 
                                       target: str,
                                       q: VectorArray,
                                       u: VectorArray) -> None:  
        for time_step in range(self.nt+1):
            self._cache_operators(
                target = target,
                time_step = time_step,
                q = q,
                u = u
            )
      
    def cache_operators(self, 
                        target: str,
                        q: VectorArray,
                        u: VectorArray = None) -> None:
        
        assert target in self._cached_operators.keys()
        assert q in self.Q
        if len(self._cached_operators['q']) != 0:
            assert np.all((self._cached_operators['q']-q).norm() == 0)
        
        if target == 'B_u':
            assert u
            assert len(u) == (self.nt + 1)

        self.logger.debug(f'Caching {target}')
        self._cached_operators['q'] = q.copy()
        

        if self.q_time_dep or (target == 'B_u'):
            self._cache_time_depended_operators(
                q = q,
                u = u,
                target = target
            )
        else:
            self._cache_operators(
                target = target,
                time_step = 0,
                q = q,
                u = u
            )
            
    def reset_cached_operators(self,
                               keys: List[str] = None) -> None:
        self.logger.debug('Deleting cache')

        if not keys:
            keys = self._cached_operators.keys()

        for key in keys:
            if key in self._cached_operators.keys():
                del self._cached_operators[key][:]
            
            if key == 'q':
                self._cached_operators['q'] = self.Q.empty()
                continue
            
            if self.q_time_dep or (key == 'B_u'):
                self._cached_operators[key] = [None] * (self.nt + 1)
            else:
                self._cached_operators[key] = [None]
    
    def update_cache(self,
                     q: VectorArray,
                     u: VectorArray = None,
                     use_cached_operators: bool = True,
                     required_cache_keys: List[str] = []):
        
        if use_cached_operators:
            if self._cache_update_required(q):
                self.reset_cached_operators()
            
            for key in required_cache_keys:
                lst = self._cached_operators[key]
                assert all(x is None for x in lst) or all(x is not None for x in lst)
                if all(x is None for x in self._cached_operators[key]):
                    self.cache_operators(q=q, u=u, target=key)
        
#%% solve methods
    def solve_state(self, 
                    q: VectorArray,
                    use_cached_operators: bool = False) -> VectorArray:
        
        assert q in self.Q

        if self.q_time_dep:
            assert len(q) == (self.nt + 1)
        else:
            assert len(q) == 1

        self.num_calls['solve_state'] += 1

        required_cache_keys = ['A_q'] 
        required_cache_keys += self.time_stepper.required_cache_keys.copy()
        self.update_cache(
            q = q, 
            use_cached_operators = use_cached_operators, 
            required_cache_keys = required_cache_keys
        )

        iterator = self.time_stepper.iterate(initial_data = self.initial_data['state'], 
                                             q=q,
                                             rhs=self.L,
                                             use_cached_operators=use_cached_operators,
                                             cached_operators=self._cached_operators)
        
        u = self.V.empty(reserve= (self.nt + 1))
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
            assert len(q) == self.nt + 1
        else:
            assert len(q) == 1

        assert len(u) == self.nt + 1

        self.num_calls['solve_adjoint'] += 1

        required_cache_keys = ['A_q'] 
        required_cache_keys += self.time_stepper.required_cache_keys.copy()
        self.update_cache(
            q = q, 
            use_cached_operators = use_cached_operators, 
            required_cache_keys = required_cache_keys)

        rhs = self.bilinear_cost_term.apply(u) - self.linear_cost_term
        rhs = self.A.clear_rhs_boundary_dofs(
            rhs = rhs,
            flip = True
        )
        
        # if isinstance(self.A, FOMEvaluatorA):
        #     # TODO The state depended parts has already zero BVs. 
        #     # Maybe zero the other part only once.
        #     rhs =  self.A.clear_rhs_boundary_dofs(
        #         rhs = rhs,
        #         flip = True
        #     )
        # else:
        #     raise NotImplementedError
            # TODO Write clear_rhs_boundary_dofs for reaction-diffusion
            # self.A.clear_rhs_boundary_dofs(rhs)
            # I = self.A.boundary_info.dirichlet_boundaries(2)
            # rhs[:,I] = 0
            # rhs = np.flip(rhs.to_numpy(), axis=0)
            # rhs = self.V.make_array(rhs)
        


        rhs = (-1) * rhs
        iterator = self.time_stepper.iterate(initial_data = self.initial_data['adjoint'], 
                                             q=q,
                                             rhs=rhs,
                                             use_cached_operators=use_cached_operators,
                                             cached_operators=self._cached_operators,
                                             config={
                                                 'implicit_euler_rhs' : True
                                             })
        
        p = self.V.empty(reserve = (self.nt + 1))
        for p_n, _ in iterator:
            p.append(p_n)

        #return self.V.make_array(np.flip(p.to_numpy(), axis=0))
        return self.A.flip_vector_array(p)
    
    def solve_linearized_state(self,
                               q: VectorArray,
                               d: VectorArray,
                               u: VectorArray,
                               use_cached_operators: bool = False) -> VectorArray:
        
        assert q in self.Q
        assert d in self.Q
        assert u in self.V
        if self.q_time_dep:
            assert len(q) == self.nt + 1
        else:
            assert len(q) == 1
        assert len(d) == len(q)
        assert len(u) == self.nt + 1

        self.num_calls['solve_linearized_state'] += 1
        
        required_cache_keys = ['A_q']
        required_cache_keys += self.time_stepper.required_cache_keys.copy()
        required_cache_keys += ['B_u']
        self.update_cache(
            q = q, 
            u = u,
            use_cached_operators = use_cached_operators, 
            required_cache_keys = required_cache_keys
        )

        if use_cached_operators:
            B_u = self._cached_operators['B_u']
        else:
            B_u = [self.B(u[idx], idx) for idx in range(len(u))]

        if self.q_time_dep:
            rhs = self.V.make_array([B_u[idx].B_u(d[idx]) for idx in range(len(u))])
        else:   
            rhs = self.V.make_array([B_u[idx].B_u(d[0]) for idx in range(len(u))])
            
        rhs = (-1) * rhs    
        iterator = self.time_stepper.iterate(initial_data = self.initial_data['lin_state'], 
                                             q=q,
                                             rhs=rhs,
                                             use_cached_operators=use_cached_operators,
                                             cached_operators=self._cached_operators)
        
        lin_u = self.V.empty(reserve= (self.nt + 1))
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
            assert len(q) == self.nt + 1
        else:
            assert len(q) == 1
        assert len(u) == self.nt + 1
        assert len(lin_u) == self.nt + 1

        self.num_calls['solve_linearized_adjoint'] += 1

        required_cache_keys = ['A_q']
        required_cache_keys += self.time_stepper.required_cache_keys.copy()        
        self.update_cache(
            q = q, 
            use_cached_operators = use_cached_operators, 
            required_cache_keys = required_cache_keys
        )

        rhs = self.bilinear_cost_term.apply(u + lin_u) - self.linear_cost_term
        rhs = (-1) * rhs
        rhs = self.A.clear_rhs_boundary_dofs(
            rhs = rhs,
            flip = True
        )

        # if isinstance(self.A, FOMEvaluatorA):
        #     # TODO The state depended parts has already zero BVs. 
        #     # Maybe zero the other part only once.
        #     rhs = self.A.clear_rhs_boundary_dofs(
        #         rhs = rhs,
        #         flip = True
        #     )
        # else:
        #     raise NotImplementedError

        # rhs = np.flip(rhs.to_numpy(), axis=0)
        # if isinstance(self.A, FOMEvaluatorA):
        #     I = self.A.boundary_info.dirichlet_boundaries(2)
        #     rhs[:,I] = 0

        # #rhs = self.delta_t * self.V.make_array(rhs)
        # rhs = self.V.make_array(rhs)#


        iterator = self.time_stepper.iterate(initial_data = self.initial_data['lin_adjoint'], 
                                            q=q,
                                            rhs=rhs,
                                            use_cached_operators=use_cached_operators,
                                            cached_operators=self._cached_operators,
                                            config={
                                                 'implicit_euler_rhs' : True
                                             })
        
        lin_p = self.V.empty(reserve= (self.nt + 1))
        for lin_p_n, _ in iterator:
            lin_p.append(lin_p_n)
        
        lin_p = self.A.flip_vector_array(lin_p)
        return lin_p


#%% objective and gradient
    def objective(self, 
                  u: Union[VectorArray, np.ndarray],
                  q: VectorArray = None,
                  alpha: float = 0) -> float:
        
        if q:
            assert q in self.Q
            if self.q_time_dep:
                assert len(q) == self.nt + 1
            else:
                assert len(q) == 1

        assert len(u) == self.nt + 1
        assert u in self.V
        assert self.bilinear_cost_term 
        assert self.linear_cost_term
        
        self.num_calls['objective'] += 1
        out = 0.5 * self.delta_t * np.sum(self.bilinear_cost_term.pairwise_apply2(u,u)
                                          + (-2) * self.linear_cost_term.pairwise_inner(u) 
                                          + self.constant_cost_term)
        if alpha > 0:
            assert q is not None
            # add regularization term if alpha >0
            # print(out)
            # print(alpha * self.regularization_term(q))
            return out + alpha * self.regularization_term(q)
        else:
            return out

    def gradient(self,
                 u: VectorArray,
                 p: VectorArray,
                 q: VectorArray = None,
                 alpha: float = 0,
                 use_cached_operators: bool = False,
                 return_per_time_step : bool = False) -> NumpyVectorArray | Tuple[NumpyVectorArray, NumpyVectorArray]:
        
        assert u in self.V
        assert p in self.V
        assert len(u) == self.nt + 1
        assert len(p) == self.nt + 1

        required_cache_keys = ['B_u']
        self.update_cache(
            q, 
            u,
            use_cached_operators, 
            required_cache_keys
        )


        if use_cached_operators:
            B_u = self._cached_operators['B_u']
        else:
            B_u = [self.B(u[idx], idx) for idx in range(len(u))]

        self.num_calls['gradient'] += 1
        grad = self.Q.empty(reserve=(self.nt + 1))

        # TODO Check if this is efficent and / or how its efficeny can be improved
        for idx in range(0, self.nt + 1):
            grad.append(self.Q.make_array(B_u[idx].B_u_ad(p[idx])))

        if not self.q_time_dep:
            _grad = self.delta_t * self.Q.make_array(np.sum(grad.to_numpy(), axis=0, keepdims=True))
        
        if self.riesz_rep_grad:
            _grad = self.products['prod_Q'].apply_inverse(_grad) 
        
        if alpha > 0:
            out = _grad + alpha * self.gradient_regularization_term(q)
        else:
            out = _grad

        if return_per_time_step:
            return (out, grad)
        else: 
            return out

    def linearized_objective(self,
                            q: VectorArray,
                            d: VectorArray,
                            u: VectorArray,
                            lin_u: VectorArray,
                            alpha : float,
                            use_cached_operators: bool = False) -> float:

        if self.q_time_dep:
            assert len(q) == self.nt + 1
        else:
            assert len(q) == 1
        assert len(d) == len(q)
        assert len(u) == self.nt + 1
        assert len(lin_u) == self.nt + 1

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
                      (-2)  * self.linear_cost_term.pairwise_inner(u_q_d) + \
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
            assert len(q) == self.nt + 1
        else:
            assert len(q) == 1
        assert len(q) == len(d)
        assert len(u) == self.nt + 1
        assert len(lin_p) == self.nt + 1
        

        assert q in self.Q
        assert d in self.Q
        assert u in self.V
        assert lin_p in self.V

        required_cache_keys = ['A_q']
        required_cache_keys += ['B_u']
        self.update_cache(
            q = q, 
            u = u,
            use_cached_operators = use_cached_operators, 
            required_cache_keys = required_cache_keys
        )

        if use_cached_operators:
            B_u = self._cached_operators['B_u']
        else:
            B_u = [self.B(u[idx], idx) for idx in range(len(u))]

        self.num_calls['linearized_gradient'] += 1
        grad = self.Q.empty(reserve=(self.nt + 1))

        #print(np.max(np.abs(grad.to_numpy())))
        # TODO Check if this is efficent and / or how its efficeny can be improved
        for idx in range(0, self.nt + 1):
            buf = self.Q.make_array(B_u[idx].B_u_ad(lin_p[idx]))
            grad.append(buf)

            # print(idx)
            # print(grad[idx])

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
            assert len(q) == self.nt + 1
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
            assert len(q) == self.nt + 1
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
            assert len(q) == self.nt + 1
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
            assert len(q) == self.nt + 1
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
        
        assert len(u) == self.nt + 1
        assert q in self.Q
        assert u in self.V

        required_cache_keys = ['A_q']
        required_cache_keys += ['residual_A_q']
        self.update_cache(
            q = q, 
            use_cached_operators = use_cached_operators, 
            required_cache_keys = required_cache_keys
        )

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
        
        assert len(u) == self.nt + 1
        assert len(u) == len(p)

        assert q in self.Q
        assert u in self.V
        assert p in self.V

        required_cache_keys = ['A_q']
        required_cache_keys += ['residual_A_q']
        self.update_cache(
            q = q, 
            use_cached_operators = use_cached_operators, 
            required_cache_keys = required_cache_keys
        )
            
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
                                 use_cached_operators: bool = False) -> float | None:

        if not self.objective_error_estimator:
            return None
        
        required_cache_keys = ['A_q']
        required_cache_keys += ['residual_A_q']
        self.update_cache(
            q = q, 
            use_cached_operators = use_cached_operators, 
            required_cache_keys = required_cache_keys
        )

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
    
    def compute_sparsity(self,
                         q: VectorArray,
                         use_cached_operators: bool = False) -> Tuple[int,float]:

        required_cache_keys = ['A_q']
        self.update_cache(
            q = q, 
            use_cached_operators = use_cached_operators, 
            required_cache_keys = required_cache_keys
        )

        A_q = self._cached_operators['A_q'][0]
        A_q = A_q.assemble()
        #try:
        nzz = A_q.matrix.nzz
        total_elements = A_q.shape[0] * A_q.shape[1]
        sparsity = 1 - (nzz / total_elements)
        return nzz, sparsity
        # except:
        #     return np.nan, np.nan
            
        
