from numbers import Number
import numpy as np

from pymor.core.base import ImmutableObject, abstractmethod
from pymor.operators.constructions import IdentityOperator, VectorArrayOperator, ZeroOperator, VectorOperator
from pymor.operators.interface import Operator
from pymor.parameters.base import Mu
from pymor.tools import floatcmp
from pymor.vectorarrays.interface import VectorArray
from pymor.algorithms.timestepping import TimeStepper, _depends_on_time
from evaluators import A_evaluator

class ImplicitEulerTimeStepper(TimeStepper):

    steps = 1

    def __init__(self, nt, solver_options='operator'):
        self.__auto_init(locals())

    def estimate_time_step_count(self, initial_time, end_time):
        return self.nt
    
    # TODO rename mu to q or remove mu
    def iterate(self, initial_time, end_time, initial_data, q, operator, rhs=None, mass=None, num_values=None):
        A, F, M, U0, t0, t1, nt = operator, rhs, mass, initial_data, initial_time, end_time, self.nt

        mu = None
        
        if isinstance(A, Operator):
            pass
        elif isinstance(A, A_evaluator):
            A = A(q)
        else:
            # Should never happend
            raise AttributeError
        
        assert isinstance(F, (type(None), Operator, VectorArray))
        assert isinstance(M, (type(None), Operator))
        
        assert A.source == A.range


        num_values = num_values or nt + 1
        dt = (t1 - t0) / nt
        DT = (t1 - t0) / (num_values - 1)

        if F is None:
            F_time_dep = False
        elif isinstance(F, Operator):
            assert F.source.dim == 1
            assert F.range == A.range
            F_time_dep = _depends_on_time(F, mu)
            if not F_time_dep:
                dt_F = F.as_vector(mu) * dt
        elif isinstance(F, VectorArray):
            assert F in A.range
            if len(F) == 1:
                F_time_dep = False
            elif len(F) == (self.nt + 1):
                F_time_dep = True
            else: 
                # Should never happend
                raise AttributeError
        else:
            # Should never happend
            raise AttributeError

        if M is None:
            from pymor.operators.constructions import IdentityOperator
            M = IdentityOperator(A.source)

        assert A.source == M.source == M.range
        
        assert not M.parametric
        assert U0 in A.source
        assert len(U0) == 1

        num_ret_values = 1
        yield U0, t0

        options = (A.solver_options if self.solver_options == 'operator' else
                   M.solver_options if self.solver_options == 'mass' else
                   self.solver_options)
        
        M_dt_A = (M + A * dt).with_(solver_options=options)
        if not _depends_on_time(M_dt_A, mu):
            M_dt_A = M_dt_A.assemble(mu)

        t = t0
        U = U0.copy()
        if mu is None:
            mu = Mu()

        for n in range(nt):
            t += dt
            mu = mu.with_(t=t)
            _rhs = M.apply(U)
            if F_time_dep:
                if isinstance(F, Operator):
                    dt_F = F.as_vector(mu) * dt
                elif isinstance(F, VectorArray):
                    dt_F = F[n] * dt
                else: 
                    # Should never happend
                    raise AttributeError

            if dt_F:
                rhs = _rhs + dt_F
            else:
                rhs = _rhs
            U = M_dt_A.apply_inverse(rhs, mu=mu, initial_guess=U)
            while t - t0 + (min(dt, DT) * 0.5) >= num_ret_values * DT:
                num_ret_values += 1
                yield U, t



