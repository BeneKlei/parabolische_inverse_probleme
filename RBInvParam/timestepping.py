
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Generator, Tuple
from enum import Enum

import numpy as np
import sys

from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.interface import VectorSpace
from pymor.tools.floatcmp import float_cmp_all

from RBInvParam.evaluators import EvaluatorA

import pymor_dealii_bindings as pd2
from RBInvParam.problems.shared.pymor_dealii_bindings.operator import DealIIMatrixOperator

class TimeStepperType(Enum):
    ImplicitEulerTimeStepper = "ImplicitEulerTimeStepper"
    SecondOrderCrankNicolson = "SecondOrderCrankNicolson"
    SecondOrderCrankNicolsonLinear = "SecondOrderCrankNicolsonLinear"
    SecondOrderCrankNicolsonAdjointDTO = "SecondOrderCrankNicolsonAdjointDTO"

class TimeStepper(ABC):
    def __init__(self, 
                nt : int, 
                M : Operator,
                A : EvaluatorA,
                Q: VectorSpace,
                V: VectorSpace,
                T_initial: float,
                T_final: float,
                q_time_dep: Dict,
                A_q_key: str = 'A_q',
                apply_adjoint: bool = True,
                key_prefix : str = '',
                config : Dict = {},
                cache_tol : float = sys.float_info.epsilon):
    
        self.nt = nt
        self.M = M 
        self.A = A
        self.Q = Q
        self.V = V
        self.T_initial = T_initial
        self.T_final = T_final
        self.q_time_dep = q_time_dep
        self.time_dep_cache_policy : Dict[str, bool] = {}
        self.required_cache_keys = []
        self.A_q_key = A_q_key
        self.apply_adjoint = apply_adjoint
        self.key_prefix = key_prefix
        self.config = config
        self.cache_tol = cache_tol

        assert isinstance(self.M, Operator)
        assert isinstance(self.A, EvaluatorA)
        assert not self.M.parametric
        assert self.A.source == self.A.range
        assert self.M.source == self.M.range
        assert self.M.range == self.A.range
        assert self.A.range == self.V
    
    def _get_A_q(self, 
                q: VectorArray,
                u: VectorArray = None) -> Operator:

        _A_q_key = self.A_q_key.replace('_ad', '')
        if _A_q_key == 'A_q':
            return self.A.get_A_q(q)
        elif _A_q_key == 'partial_q_A_q_u':
            return self.A.get_partial_q_A_q_u(q, u)
        elif _A_q_key == 'partial_u_A_q_u':            
            return self.A.get_partial_u_A_q_u(q, u)
        else:
            self.logger.error(f'Unknown target {self.A_q_key}.')
            raise ValueError


    @abstractmethod
    def iterate(self,                               
                initial_data : VectorArray, 
                q : Union[VectorArray, List[VectorArray]], 
                u : Union[VectorArray, List[VectorArray]],                 
                rhs : Union[VectorArray, List[VectorArray]],
                use_cached_operators: bool = False,
                cached_operators: Dict = None,
                config: Dict = None) -> Generator[Tuple[VectorArray, float], None, None]:
        pass
    
    @abstractmethod
    def cache_operator(self,
                       target: str,
                       q : VectorArray,
                       u : VectorArray) -> Operator:
        pass

    def _check_cache(self,
                     keys: List[str], 
                     q : Union[VectorArray, List[VectorArray]],
                     cached_operators: Dict = None):
        'q' in cached_operators.keys()

        if len(cached_operators['q']) != 0:
            diff = cached_operators['q'] - q            
            assert bool((diff.norm() <= self.cache_tol).any())
        
    def _check_initial_data(self,
                            initial_data: dict,
                            key: str):

        assert key in initial_data.keys()
        data = initial_data[key]
        assert isinstance(data, VectorArray)
        assert data in self.A.source
        assert len(data) == 1

class ImplicitEulerTimeStepper(TimeStepper):
    def iterate(self,                               
                initial_data : VectorArray, 
                q : Union[VectorArray, List[VectorArray]], 
                rhs : Union[VectorArray, List[VectorArray]],
                use_cached_operators: bool = False,
                cached_operators: Dict = None) -> Generator[Tuple[VectorArray, float], None, None]:
    
        F, U0 = rhs, initial_data
        dt_F = None
        theta = self.theta
 
        assert isinstance(F, (VectorArray))
        assert isinstance(q, (VectorArray, np.ndarray))
        assert U0 in self.A.source
        assert len(U0) == 1
        assert q in self.Q

        if use_cached_operators:
            self._check_cache(
               keys = ['M_dt_A'],
               q = q,
               cached_operators = cached_operators
            )

        num_values = self.nt + 1
        dt = (self.T_final - self.T_initial) / self.nt
        DT = (self.T_final - self.T_initial) / (num_values - 1)

        if F is None:
            F_time_dep = False
        elif isinstance(F, VectorArray):
            assert F in self.A.range
            if len(F) == 1:
                F_time_dep = False
                dt_F = F * dt
            elif len(F) == (self.nt):
                F_time_dep = True
            else: 
                # Should never happend
                raise AttributeError
        else:
            # Should never happend
            raise AttributeError
    
        num_ret_values = 1
        M_dt_A_q = None    
        
        if use_cached_operators:
            M_dt_A_q = cached_operators['M_dt_A_q'][0]
        else:
            A_q = self.A(q[0])
            M_dt_A_q = (self.M + A_q * dt).assemble()
            
        t = self.T_initial
        U = U0.copy()

        for n in range(self.nt):
            t += dt

            _rhs = self.M.apply(U)

            if F_time_dep:
                if isinstance(F, VectorArray):
                    dt_F = F[n] * dt
                else: 
                    # Should never happend
                    raise AttributeError
                
            if self.q_time_dep:
                if use_cached_operators:
                    M_dt_A_q = cached_operators['M_dt_A_q'][n]
                else:
                    A_q = self.A(q[n])
                    M_dt_A_q = (self.M + A_q * dt)
                
            assert M_dt_A_q is not None

            if dt_F:
                rhs = _rhs + dt_F
            else:
                rhs = _rhs
            

            U = M_dt_A_q.apply_inverse(rhs, initial_guess=U)

            while t - self.T_initial + (min(dt, DT) * 0.5) >= num_ret_values * DT:
                num_ret_values += 1
                yield U, t

class SecondOrderCrankNicolson(TimeStepper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert 'zeta' in self.config.keys()
        zeta = self.config['zeta']

        assert 0 <= zeta <= 1 
        self.zeta = zeta

        _A_q_key = self.A_q_key.replace('_ad', '')
        if _A_q_key == 'A_q':
            policy = self.q_time_dep
        elif _A_q_key == 'partial_q_A_q_u':
            policy = True
        elif _A_q_key == 'partial_u_A_q_u':
            policy = not self.A.A_q_linear or self.q_time_dep
        else:
            self.logger.error(f'Unknown target {self.A_q_key}.')
            raise ValueError

        self.time_dep_cache_policy = {
            self.key_prefix + '_' + 'S_zeta' : policy,
            self.key_prefix + '_' + 'S_zeta_minus_one' : policy
        }

        self.required_cache_keys = list(self.time_dep_cache_policy.keys())
    
    def cache_operator(self,
                       target: str,
                       time_step: int,
                       q: VectorArray,
                       u: VectorArray,
                       A_q: Operator) -> Operator:
        
        if self.q_time_dep:
            assert time_step == 0

        zeta = self.zeta 
        dt = (self.T_final - self.T_initial) / self.nt

        if target == (self.key_prefix + '_' +'S_zeta'):
            return (self.M + dt**2 * zeta**2 * A_q).assemble()
        elif target == (self.key_prefix + '_' + 'S_zeta_minus_one'):
            return (self.M + dt**2 * zeta * (zeta - 1) * A_q).assemble()
        else:
            raise ValueError

    def iterate(self,                               
                initial_data : dict, 
                q : Union[VectorArray, List[VectorArray]], 
                rhs : VectorArray,
                u : Union[VectorArray, List[VectorArray]] = None,
                use_cached_operators: bool = False,
                cached_operators: Dict = None,
                config: Dict = None) -> Generator[Tuple[VectorArray, float], None, None]:
        
        ################################### Prepare ###################################
        # assert q in self.Q

        # if self.q_time_dep:
        #     assert len(q) == self.nt + 1
        # else:
        #     assert len(q) == 1

        # assert isinstance(rhs, VectorArray)
        # assert len(rhs) in (self.nt + 1, 1)

        # if u: 
        #     assert u in self.V
        #     assert len(u) == self.nt + 1

        if len(rhs) == 1:
            rhs_time_dep = False
        else:
            rhs_time_dep = True

        implicit_euler_rhs = False
        if config and config['implicit_euler_rhs']:
            implicit_euler_rhs = config['implicit_euler_rhs']

        for key in ['zeroth_order', 'first_order']:
            self._check_initial_data(initial_data, key)

        if use_cached_operators:
            self._check_cache(
               keys = self.required_cache_keys,
               q = q,
               cached_operators = cached_operators
            )
                
        num_values = self.nt + 1
        dt = (self.T_final - self.T_initial) / self.nt
        DT = (self.T_final - self.T_initial) / (num_values - 1)

        zeta = self.zeta

        ################################### First step ###################################
        U_cur = initial_data['zeroth_order']
        U_dot_cur = initial_data['first_order']
        if not self.apply_adjoint:
            M_U_dot_cur = self.M.apply(U_dot_cur)
        else:
            M_U_dot_cur = self.M.apply_adjoint(U_dot_cur)

        t = self.T_initial
        yield U_cur, U_dot_cur, t

        rhs_pre = rhs[0]
        rhs_cur = rhs[0]

        dt_R = rhs[0].copy()
        dt_R.scal(0.0)

        if not rhs_time_dep:
            dt_R.axpy(dt, rhs_cur)   # dt_R = dt * rhs
                
        if use_cached_operators:
            A_q = cached_operators[self.A_q_key][0]
            S_zeta = cached_operators[(self.key_prefix + '_' + 'S_zeta')][0]
            S_zeta_minus_one = cached_operators[(self.key_prefix + '_' + 'S_zeta_minus_one')][0]
        else: 
            A_q = self._get_A_q(q[0], u[0] if u else None)
            S_zeta = self.M + dt**2 * zeta**2 * A_q
            S_zeta_minus_one = self.M + dt**2 * zeta * (zeta - 1) * A_q


        A_q = A_q.assemble()
        S_zeta = S_zeta.assemble()
        S_zeta_minus_one = S_zeta_minus_one.assemble()


        ################################### Stepping ###################################

        for n in range(1,self.nt+1):
            t += dt

            # NOTE: U_pre does not need a copy (U_cur is not modified in-place before it's no longer needed)
            U_pre = U_cur.copy()
            M_U_dot_pre = M_U_dot_cur.copy()

            # ---- RHS combination (no allocations) ----
            if rhs_time_dep:
                rhs_pre = rhs_cur
                rhs_cur = rhs[n]

                dt_R.scal(0.0)
                if implicit_euler_rhs:
                    dt_R.axpy(1.0, rhs_cur)
                else:
                    dt_R.axpy(zeta, rhs_cur)
                    dt_R.axpy(1.0 - zeta, rhs_pre)
                dt_R.scal(dt)
            
            # ---- operator update if q_time_dep ----
            if self.q_time_dep:
                # Otherwise the values set above are never updated
                if use_cached_operators:
                    A_q = cached_operators[self.A_q_key][n]
                    S_zeta = cached_operators[(self.key_prefix + '_' + 'S_zeta')][n]
                    S_zeta_minus_one = cached_operators[(self.key_prefix + '_' + 'S_zeta_minus_one')][n]
                else:
                    A_q = self._get_A_q(q[0], u[0] if u else None)
                    S_zeta = self.M + dt**2 * zeta**2 * A_q
                    S_zeta_minus_one = self.M + dt**2 * zeta * (zeta - 1) * A_q
                
                A_q = A_q.assemble()
                S_zeta = S_zeta.assemble()
                S_zeta_minus_one = S_zeta_minus_one.assemble()


            # --------------------------------------------------------------
            _lhs = S_zeta
            _rhs = self.M.apply(U_pre)
            _rhs.axpy(zeta * dt, M_U_dot_pre)
            _rhs.axpy(zeta * zeta, dt_R)

            if not self.apply_adjoint:
                _U = _lhs.apply_inverse(_rhs)
                assert np.max(np.abs(_lhs.apply(_U).to_numpy()-_rhs.to_numpy())) <= 1e-12
            else:
                _U = _lhs.apply_inverse_adjoint(_rhs)
                assert np.max(np.abs(_lhs.apply_adjoint(_U).to_numpy()-_rhs.to_numpy())) <= 1e-12


            # --------------------------------------------------------------
            if not self.apply_adjoint:
                A_q_U = A_q.apply(_U)
            else:
                A_q_U = A_q.apply_adjoint(_U)

            M_U_dot_cur.scal(0.0)
            M_U_dot_cur.axpy(1.0, M_U_dot_pre)
            M_U_dot_cur.axpy(-dt, A_q_U)
            M_U_dot_cur.axpy(1.0, dt_R)

            if not self.apply_adjoint:
                U_dot_cur = self.M.apply_inverse(M_U_dot_cur)
            else:
                U_dot_cur = self.M.apply_inverse_adjoint(M_U_dot_cur)

            # --------------------------------------------------------------
            U_cur = _U
            U_cur.axpy(-(1.0 - zeta), U_pre)
            U_cur.scal(1.0 / zeta)

            yield U_cur, U_dot_cur, t

# TODO Thats a ylegacy version of CN timestepper for linear operators ONLY. Should be merged or replaced by the more general CN timestepper
class SecondOrderCrankNicolsonLinear(TimeStepper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert 'zeta' in self.config.keys()
        zeta = self.config['zeta']

        assert 0 <= zeta <= 1 
        self.zeta = zeta
        self.required_cache_keys = [
            self.key_prefix + '_' + 'S_zeta', 
            self.key_prefix + '_' + 'S_zeta_minus_one'
        ]
    
    def cache_operator(self,
                       target: str,
                       time_step: int,
                       q : VectorArray,
                       u : VectorArray,
                       A_q: Operator) -> Operator:
        
        if self.q_time_dep:
            assert time_step == 0

        zeta = self.zeta 
        dt = (self.T_final - self.T_initial) / self.nt

        if target == (self.key_prefix + '_' +'S_zeta'):
            return self.M + dt**2 * zeta**2 * A_q
        elif target == (self.key_prefix + '_' + 'S_zeta_minus_one'):
            return self.M + dt**2 * zeta * (zeta - 1) * A_q
        else:
            raise ValueError

    def iterate(self,                               
                initial_data : dict, 
                q : Union[VectorArray, List[VectorArray]], 
                rhs : VectorArray,
                use_cached_operators: bool = False,
                cached_operators: Dict = None,
                config: Dict = None) -> Generator[Tuple[VectorArray, float], None, None]:
        
        ################################### Prepare ###################################
        assert isinstance(rhs, VectorArray)
        assert len(rhs) in (self.nt + 1, 1)

        if len(rhs) == 1:
            rhs_time_dep = False
        else:
            rhs_time_dep = True

        assert isinstance(q, (VectorArray, np.ndarray))
        assert q in self.Q

        implicit_euler_rhs = False
        if config and config['implicit_euler_rhs']:
            implicit_euler_rhs = config['implicit_euler_rhs']

        for key in ['zeroth_order', 'first_order']:
            self._check_initial_data(initial_data, key)

        if use_cached_operators:
            self._check_cache(
               keys = self.required_cache_keys,
               q = q,
               cached_operators = cached_operators
            )
                
        num_values = self.nt + 1
        dt = (self.T_final - self.T_initial) / self.nt
        DT = (self.T_final - self.T_initial) / (num_values - 1)

        zeta = self.zeta

        ################################### First step ###################################

        U_cur = initial_data['zeroth_order']
        U_dot_cur = initial_data['first_order']
        if not self.apply_adjoint:
            M_U_dot_cur = self.M.apply(U_dot_cur)
        else:
            M_U_dot_cur = self.M.apply_adjoint(U_dot_cur)

        t = self.T_initial
        yield U_cur, U_dot_cur, t

        U_pre = U_cur.copy()
        M_U_dot_pre = M_U_dot_cur.copy()

        rhs_pre = rhs[0].copy()
        rhs_cur = rhs[0].copy()

        # TODO implement non linear operator

        if use_cached_operators:
            A_q = cached_operators[self.A_q_key][0]
            S_zeta = cached_operators[(self.key_prefix + '_' + 'S_zeta')][0]
            S_zeta_minus_one = cached_operators[(self.key_prefix + '_' + 'S_zeta_minus_one')][0]
        else:
            A_q = self.A.get_A_q(q[0])
            S_zeta = self.M + dt**2 * zeta**2 * A_q
            S_zeta_minus_one = self.M + dt**2 * zeta * (zeta - 1) * A_q
        
        assert A_q.linear
        A_q = A_q.assemble()
        S_zeta = S_zeta.assemble()
        S_zeta_minus_one = S_zeta_minus_one.assemble()

        if not rhs_time_dep:
            dt_R = dt * rhs

        ################################### Stepping ###################################

        for n in range(1,self.nt+1):
            t += dt
            U_pre = U_cur
            M_U_dot_pre = M_U_dot_cur

            if rhs_time_dep:
                rhs_pre = rhs_cur
            
            if self.q_time_dep:
                # Otherwise the values set above are never updated
                if use_cached_operators:
                    A_q = cached_operators[self.A_q_key][n]
                    S_zeta = cached_operators[(self.key_prefix + '_' + 'S_zeta')][n]
                    S_zeta_minus_one = cached_operators[(self.key_prefix + '_' + 'S_zeta_minus_one')][n]
                else:
                    A_q = self.A.get_A_q(q[n])
                    S_zeta = self.M + dt**2 * zeta**2 * A_q
                    S_zeta_minus_one = self.M + dt**2 * zeta * (zeta - 1) * A_q
                
                A_q = A_q.assemble()
                S_zeta = S_zeta.assemble()
                S_zeta_minus_one = S_zeta_minus_one.assemble()

            if rhs_time_dep:#
                rhs_cur = rhs[n]

                if implicit_euler_rhs:
                    dt_R = rhs_cur
                else:
                    dt_R = zeta * rhs_cur
                    dt_R += (1.0 - zeta) * rhs_pre
                
                dt_R *= dt

            # --------------------------------------------------------------
            _lhs = S_zeta
            if not self.apply_adjoint:
                _rhs = S_zeta_minus_one.apply(U_pre)
                
            else:
                _rhs = S_zeta_minus_one.apply_adjoint(U_pre)                

            _rhs += dt * M_U_dot_pre
            _rhs += zeta * dt_R

            if not self.apply_adjoint:
                # TODO rework s.t. the the deal.ii solver is used
                U_cur = _lhs.apply_inverse(_rhs)
                #assert np.max(np.abs(_lhs.apply(U_cur).to_numpy()-_rhs.to_numpy())) <= 1e-12
            else:
                U_cur = _lhs.apply_inverse_adjoint(_rhs)
                #assert np.max(np.abs(_lhs.apply_adjoint(U_cur).to_numpy()-_rhs.to_numpy())) <= 1e-12


            # --------------------------------------------------------------
            M_U_dot_cur = M_U_dot_pre
            _U = zeta * U_cur + (1 - zeta) * U_pre
            if not self.apply_adjoint:
                A_q_U = A_q.apply(_U)
            else:
                A_q_U = A_q.apply_adjoint(_U)

            M_U_dot_cur += (-1) * dt * A_q_U
            M_U_dot_cur += dt_R

            if not self.apply_adjoint:
                U_dot_cur = self.M.apply_inverse(M_U_dot_cur)
            else:
                U_dot_cur = self.M.apply_inverse_adjoint(M_U_dot_cur)

            # --------------------------------------------------------------

            yield U_cur, U_dot_cur, t


class SecondOrderCrankNicolsonAdjointDTO(SecondOrderCrankNicolson):
    def iterate(self,                               
            initial_data : dict, 
            q : Union[VectorArray, List[VectorArray]], 
            rhs : VectorArray,
            use_cached_operators: bool = False,
            cached_operators: Dict = None,
            config: Dict = None) -> Generator[Tuple[VectorArray, float], None, None]:
        
        ################################### Prepare ###################################
        assert isinstance(rhs, VectorArray)
        assert len(rhs) in (self.nt + 1, 1)

        if len(rhs) == 1:
            rhs_time_dep = False
        else:
            rhs_time_dep = True

        assert isinstance(q, (VectorArray, np.ndarray))
        assert q in self.Q

        for key in ['zeroth_order', 'first_order']:
            self._check_initial_data(initial_data, key)

        if use_cached_operators:
            self._check_cache(
               keys = self.required_cache_keys,
               q = q,
               cached_operators = cached_operators
            )
                
        num_values = self.nt + 1
        dt = (self.T_final - self.T_initial) / self.nt
        DT = (self.T_final - self.T_initial) / (num_values - 1)

        zeta = self.zeta

        ################################### First step ###################################

        U_cur = initial_data['zeroth_order']
        U_dot_cur = initial_data['first_order']
        M_U_dot_cur = self.M.apply(U_dot_cur)

        t = self.T_initial
        yield U_cur, U_dot_cur, t

        U_pre = U_cur.copy()
        M_U_dot_pre = M_U_dot_cur.copy()

        rhs_pre = rhs[0].copy()
        rhs_cur = rhs[0].copy()

        if use_cached_operators:
            A_q = cached_operators[self.A_q_key][0]
            S_zeta = cached_operators[(self.key_prefix + '_' + 'S_zeta')][0]
            S_zeta_minus_one = cached_operators[(self.key_prefix + '_' + 'S_zeta_minus_one')][0]
        else:
            A_q = self.A.get_A_q(q[0])
            S_zeta = self.M + dt**2 * zeta**2 * A_q
            S_zeta_minus_one = self.M + dt**2 * zeta * (zeta - 1) * A_q


        A_q = A_q.assemble()
        S_zeta = S_zeta.assemble()
        S_zeta_minus_one = S_zeta_minus_one.assemble()
        
        if not rhs_time_dep:
            dt_R = dt * rhs

        ################################### Stepping ###################################

        for n in range(1,self.nt+1):
            t += dt
            U_pre = U_cur
            M_U_dot_pre = M_U_dot_cur
            U_dot_pre = U_dot_cur

            if rhs_time_dep:
                rhs_pre = rhs_cur
            
            if self.q_time_dep:
                # Otherwise the values set above are never updated
                if use_cached_operators:
                    A_q = cached_operators[self.A_q_key][n]
                    S_zeta = cached_operators[(self.key_prefix + '_' + 'S_zeta')][n]
                    S_zeta_minus_one = cached_operators[(self.key_prefix + '_' + 'S_zeta_minus_one')][n]
                else:
                    A_q = self.A.get_A_q(q[n])
                    S_zeta = self.M + dt**2 * zeta**2 * A_q
                    S_zeta_minus_one = self.M + dt**2 * zeta * (zeta - 1) * A_q

            if rhs_time_dep:
                dt_R = rhs[n]              
                dt_R *= dt

            # --------------------------------------------------------------

            M_U_dot_cur = M_U_dot_pre
            if not self.apply_adjoint:
                M_U_pre = self.M.apply(U_pre)
            else:
                M_U_pre = self.M.apply_adjoint(U_pre)

            M_U_dot_cur += dt * M_U_pre
            if not self.apply_adjoint:
                U_dot_cur = self.M.apply_inverse(M_U_dot_cur)
            else:
                U_dot_cur = self.M.apply_inverse_adjoint(M_U_dot_cur)

            # --------------------------------------------------------------
            _U_dot = zeta * U_dot_cur + (1 - zeta) * U_dot_pre

            _lhs = S_zeta

            if not self.apply_adjoint:
                _rhs = S_zeta_minus_one.apply(U_pre)
                A_q_U_dot = A_q.apply(_U_dot)
            else:
                _rhs = S_zeta_minus_one.apply_adjoint(U_pre)
                A_q_U_dot = A_q.apply_adjoint(_U_dot)

            _rhs += (-1) * dt * A_q_U_dot
            _rhs += dt_R

            if not self.apply_adjoint:
                U_cur = _lhs.apply_inverse(_rhs)
                assert np.max(np.abs(_lhs.apply(U_cur).to_numpy()-_rhs.to_numpy())) <= 1e-12   
            else:
                U_cur = _lhs.apply_inverse_adjoint(_rhs)
                assert np.max(np.abs(_lhs.apply_adjoint(U_cur).to_numpy()-_rhs.to_numpy())) <= 1e-12   

            yield U_cur, U_dot_cur, t


def create_time_stepper(time_stepper_type: TimeStepperType,
                        **kwargs) -> TimeStepper:
    if time_stepper_type == TimeStepperType.ImplicitEulerTimeStepper:
        return ImplicitEulerTimeStepper(**kwargs)
    elif time_stepper_type == TimeStepperType.SecondOrderCrankNicolson:
        return SecondOrderCrankNicolson(**kwargs)
    elif time_stepper_type == TimeStepperType.SecondOrderCrankNicolsonLinear:
        return SecondOrderCrankNicolsonLinear(**kwargs)
    elif time_stepper_type == TimeStepperType.SecondOrderCrankNicolsonAdjointDTO:
        return SecondOrderCrankNicolsonAdjointDTO(**kwargs)
    
    raise ValueError(f"Unsupported time stepper type: {time_stepper_type}")

