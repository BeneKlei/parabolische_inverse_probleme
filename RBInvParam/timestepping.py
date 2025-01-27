from numbers import Number
import numpy as np

from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.interface import VectorSpace


from RBInvParam.evaluators import UnAssembledA, AssembledA

class ImplicitEulerTimeStepper():
    def __init__(self, 
                 nt : int, 
                 Q: VectorSpace,
                 V: VectorSpace,
                 solver_options=None):
        
        self.nt = nt
        self.Q = Q
        self.V = V
        self.solver_options = solver_options

        self.cached_operators = {
            'q' : self.Q.empty(),
            'A_q' : self.Q.empty(),
            'A_q_inv' : self.Q.empty(),
            'M_dt_A_q' : self.Q.empty(),
            'M_dt_A_q_inv' : self.Q.empty()
        }


    def cache_operators(self, 
                    q: VectorArray,
                    target: str = 'M_dt_A_q_inv') -> None:
        assert target in [
            'A_q',
            'A_q_inv',
            'M_dt_A_q',
            'M_dt_A_q_inv'
        ]
        assert q in self.Q
        if not self.cached_operators['q'].empty():
            assert q == self.cached_operators['q']

        self.cached_operators['q'] = q.copy()
        
            
    def delete_cached_operators(self) -> None:
        self.cached_operators = {
            'q' : self.Q.empty(),
            'A_q' : self.Q.empty(),
            'A_q_inv' : self.Q.empty(),
            'M_dt_A_q' : self.Q.empty(),
            'M_dt_A_q_inv' : self.Q.empty()
        }
    
    def iterate(self, 
                initial_time, 
                end_time, 
                initial_data, 
                q, 
                operator, 
                rhs, 
                mass):
    
        A, F, M, U0, t0, t1, nt = operator, rhs, mass, initial_data, initial_time, end_time, self.nt
        dt_F = None
 
        assert isinstance(F, (VectorArray))
        assert isinstance(M, (Operator))
        assert isinstance(A, (UnAssembledA, AssembledA))
        assert isinstance(q, (VectorArray, np.ndarray))
        assert not M.parametric
        assert U0 in A.source
        assert len(U0) == 1
        assert A.source == A.range
        assert M.source == M.range
        assert M.range == A.range
        assert q in self.Q
        assert A.range == self.V

        num_values = nt + 1
        dt = (t1 - t0) / nt
        DT = (t1 - t0) / (num_values - 1)

        if F is None:
            F_time_dep = False
        elif isinstance(F, VectorArray):
            assert F in A.range
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
        M_dt_A = None    
        
        A_q = A(q[0])
        M_dt_A = (M + A_q * dt).with_(solver_options=self.solver_options)
        M_dt_A = M_dt_A.assemble()

        t = t0
        U = U0.copy()

        for n in range(nt):
            t += dt

            _rhs = M.apply(U)

            if F_time_dep:
                if isinstance(F, VectorArray):
                    dt_F = F[n] * dt
                else: 
                    # Should never happend
                    raise AttributeError
                
            if q_time_dep:
                A_q = A(q[n])
                M_dt_A = (M + A_q * dt).with_(solver_options=self.solver_options)
        
            assert M_dt_A is not None

            if dt_F:
                rhs = _rhs + dt_F
            else:
                rhs = _rhs
            
            U = M_dt_A.apply_inverse(rhs, initial_guess=U)

            while t - t0 + (min(dt, DT) * 0.5) >= num_ret_values * DT:
                num_ret_values += 1
                yield U, t
