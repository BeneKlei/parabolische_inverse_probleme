from typing import Dict, Union
from numbers import Number
import numpy as np

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.interface import VectorArray
from pymor.operators.interface import Operator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from evaluators import A_evaluator, B_evaluator
from timestepping import ImplicitEulerTimeStepper

class InstationaryModelIP:
    def __init__(
        self,
        u_0 : VectorArray, 
        M : Operator,
        A : A_evaluator,
        f : VectorArray,
        B : B_evaluator,
        constant_cost_term: Number,
        linear_cost_term: NumpyMatrixOperator,
        bilinear_cost_term: NumpyMatrixOperator,
        Q_h : NumpyVectorSpace,
        q_circ: VectorArray,
        constant_reg_term: Number,
        linear_reg_term: NumpyMatrixOperator,
        bilinear_reg_term: NumpyMatrixOperator,
        products : Dict,
        visualizer,
        dims: Dict,
        model_parameter: Dict,
    ):
        self.u_0 = u_0
        self.M = M 
        self.A = A 
        self.f = f 
        self.B = B 
        self.constant_cost_term = constant_cost_term 
        self.linear_cost_term = linear_cost_term 
        self.bilinear_cost_term = bilinear_cost_term 
        self.q_circ = q_circ 
        self.constant_reg_term = constant_reg_term 
        self.linear_reg_term = linear_reg_term 
        self.bilinear_reg_term = bilinear_reg_term 
        self.products = products
        self.visualizer = visualizer
        self.dims = dims
        self.model_parameter = model_parameter

        self.timestepper = ImplicitEulerTimeStepper(
            nt = self.dims['nt']
        )

        self.V_h = M.source
        self.Q_h = Q_h

        
    def solve_state(self, q: Union[VectorArray, np.ndarray]):
        assert isinstance(q, (VectorArray, np.ndarray))

        # if isinstance(q, (np.ndarray)):
        #     q = self.Q_h.make_array(q)
             
        # assert q.space == self.Q_h
        # assert (len(q) == 1) or (len(q) == (self.dims['nt'] + 1))

        iterator = self.timestepper.iterate(initial_time = self.model_parameter['T_initial'], 
                                            end_time = self.model_parameter['T_final'], 
                                            initial_data = self.u_0, 
                                            q=q,
                                            operator = self.A, 
                                            rhs=self.f, 
                                            mass=self.M)
        
        U = self.V_h.empty(reserve= self.dims['nt'] + 1)
        for U_n, _ in iterator:
            U.append(U_n)
        return U

    def solve_adjoint(self, 
                      q: Union[VectorArray, np.ndarray], 
                      u: Union[VectorArray, np.ndarray]):
        
        assert isinstance(q, (VectorArray, np.ndarray))
        assert isinstance(u, (VectorArray, np.ndarray))

        print(self.bilinear_cost_term.apply(u).dim)
        print(len(self.bilinear_cost_term.apply(u)))
        print(self.linear_cost_term.as_range_array())
        rhs = self.bilinear_cost_term.apply(u) - self.linear_cost_term.as_range_array()
        print(type(rhs))

        import sys
        sys.exit()

        iterator = self.timestepper.iterate(initial_time = self.model_parameter['T_initial'], 
                                            end_time = self.model_parameter['T_final'], 
                                            initial_data = self.u_0, 
                                            q=q,
                                            operator = self.A, 
                                            rhs=rhs, 
                                            mass=self.M,
                                            )
        
        # TODO Shift nicht vergessen
        P = self.V_h.empty(reserve= self.dims['nt'] + 1)
        for P_n, _ in iterator:
            P.append(P_n)
        return P
        

    def objective(self):
        pass

    def gradient(self):
        pass

    def solve_linearized_state(self):
        pass

    def solve_linearized_adjoint(self):
        pass

    def linearized_objective(self):
        pass

    def linearized_gradient(self):
        pass

    def linearized_hessian(self):
        pass