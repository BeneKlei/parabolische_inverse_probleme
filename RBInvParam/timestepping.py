from typing import Dict, Union
import numpy as np

from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.interface import VectorSpace
from pymor.tools.floatcmp import float_cmp_all


from RBInvParam.evaluators import UnAssembledA, AssembledA

class ImplicitEulerTimeStepper():
    def __init__(self, 
                 nt : int, 
                 M : Operator,
                 A : Union[UnAssembledA, AssembledA],
                 Q: VectorSpace,
                 V: VectorSpace,
                 T_initial: float,
                 T_final: float,
                 setup: Dict,
                 solver_options=None):
        
        self.nt = nt
        self.M = M 
        self.A = A
        self.Q = Q
        self.V = V
        self.T_initial = T_initial
        self.T_final = T_final
        self.solver_options = solver_options        
        self.setup = setup

        self.cached_operators = {
            'q' : self.Q.empty(),
            'A_q' : [],
            'M_dt_A_q' : [],
        }

        assert isinstance(self.M, Operator)
        assert isinstance(self.A, (UnAssembledA, AssembledA))
        assert not self.M.parametric
        assert self.A.source == self.A.range
        assert self.M.source == self.M.range
        assert self.M.range == self.A.range
        assert self.A.range == self.V

    def cache_operators(self, 
                        q: VectorArray,
                        target: str = 'M_dt_A_q') -> None:
        assert target in [
            'A_q',
            'M_dt_A_q'            
        ]
        assert q in self.Q
        if len(self.cached_operators['q']) != 0:
            assert q == self.cached_operators['q']

        self.cached_operators['q'] = q.copy()
        # print("Here")
        # print(self.cached_operators['q'])

        dt = (self.T_final - self.T_initial) / self.nt
        

        if self.setup['model_parameter']['q_time_dep']:            
            for n in range(self.nt):
                A_q = self.A(q[n])
                M_dt_A_q = (self.M + A_q * dt).with_(solver_options=self.solver_options)
                
                if target == 'A_q':
                    self.cached_operators[target].append(
                        A_q.assemble()
                    )
                elif target == 'M_dt_A_q':
                    self.cached_operators[target].append(
                         M_dt_A_q.assemble()
                    )
        else:    
            A_q = self.A(q[0])
            M_dt_A_q = (self.M + A_q * dt).with_(solver_options=self.solver_options)

            if target == 'A_q':
                self.cached_operators[target].append(
                    A_q.assemble()
                )
            elif target == 'M_dt_A_q':
                self.cached_operators[target].append(
                    M_dt_A_q.assemble()        
                )

    def delete_cached_operators(self) -> None:
        self.cached_operators = {
            'q' : self.Q.empty(),
            'A_q' : [],
            'M_dt_A_q' : [],
        }
    
    def iterate(self,                               
                initial_data, 
                q, 
                rhs,
                use_cached_operators: bool = False):
    
        F, U0 = rhs, initial_data
        dt_F = None
 
        assert isinstance(F, (VectorArray))
        assert isinstance(q, (VectorArray, np.ndarray))
        assert U0 in self.A.source
        assert len(U0) == 1
        assert q in self.Q

        if use_cached_operators:
            # print("AAAAAAAAAAAAAAAAAAAAAAAa")
            # print(self.cached_operators['q'])
            # print(q)
            # print(()
            # import sys
            # sys.exit()
            assert ((self.cached_operators['q']-q).norm() <= 1e-16)[0]

            assert len(self.cached_operators['M_dt_A_q']) != 0

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
    
        
        if len(q) == 1:
            q_time_dep = False
        elif len(q) == (self.nt):
            q_time_dep = True
        else:
            # Should never happend
            raise AttributeError

        num_ret_values = 1
        M_dt_A_q = None    
        
        if use_cached_operators:
            M_dt_A_q = self.cached_operators['M_dt_A_q'][0]
        else:
            A_q = self.A(q[0])
            M_dt_A_q = (self.M + A_q * dt).with_(solver_options=self.solver_options)
            M_dt_A_q = M_dt_A_q.assemble()

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
                
            if q_time_dep:
                if use_cached_operators:
                    M_dt_A_q = self.cached_operators['M_dt_A_q'][n]
                else:
                    A_q = self.A(q[n])
                    M_dt_A_q = (self.M + A_q * dt).with_(solver_options=self.solver_options)

            assert M_dt_A_q is not None

            if dt_F:
                rhs = _rhs + dt_F
            else:
                rhs = _rhs
            
            U = M_dt_A_q.apply_inverse(rhs, initial_guess=U)

            while t - self.T_initial + (min(dt, DT) * 0.5) >= num_ret_values * DT:
                num_ret_values += 1
                yield U, t
