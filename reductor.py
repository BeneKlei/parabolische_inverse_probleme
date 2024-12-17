from typing import Dict
import numpy as np

from pymor.reductors.basic import ProjectionBasedReductor
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.parameters.base import Parameters
#from pymor.operators.interface import Operator

from model import InstationaryModelIP
from evaluators import AssembledA, AssembledB
from utils import split_constant_and_parameterized_operator

class InstationaryModelIPReductor(ProjectionBasedReductor):
    def __init__(self, 
                 FOM: InstationaryModelIP, 
                 check_orthonormality: bool =False, 
                 check_tol: float =1e-3):
        
        assert isinstance(FOM, InstationaryModelIP)
        assert 'prod_V' in products.keys()
        assert 'prod_Q' in products.keys()

        bases = {
            'state_basis' : FOM.V.empty(),
            'parameter_basis' : FOM.Q.empty()
        }

        _products = {
            'state_basis' : FOM.products['prod_V'],
            'parameter_basis' : FOM.products['prod_Q']
        }

        super().__init__(FOM, 
                         bases, 
                         _products,
                         check_orthonormality=check_orthonormality, 
                         check_tol=check_tol)
    

    def project_vectorarray(self, 
                            x : VectorArray,
                            basis: str) -> np.array:

        _basis = self.bases[basis]
        assert isinstance(x, VectorArray)

        if len(_basis) == 0:
            return x.to_numpy()
        else:
            #return _basis.lincomb(x.inner(_basis, self.products[basis]))
            return x.inner(_basis, self.products[basis])
        
    
    def _assemble_reduced_operator(self) -> LincombOperator:
        # TODO Handle basis extention via offset
        state_basis = self.bases['state_basis']
        parameter_basis = self.bases['parameter_basis']

        if len(parameter_basis) == 0:
            raise NotImplementedError

        if len(state_basis) == 0:
            state_basis = None

        assert len(self.FOM.model_parameter['parameters']) == 1
        parameter_name = list(self.FOM.model_parameter['parameters'].keys())[0] 

        operators = [project(self.FOM.A.constant_operator, 
                             state_basis, 
                             state_basis)]
        coefficients = [1]

        for i in range(len(parameter_basis)):
            q_i = parameter_basis[i].to_numpy()[0]
            A_q = self.FOM.A._assemble_A_q(q_i)
            A_q = NumpyMatrixOperator(A_q,
                                      source_id = self.FOM.A.source.id, 
                                      range_id = self.FOM.A.range.id)
            A_q = project(A_q, state_basis, state_basis)
            operators.append(A_q)
            coefficients.append(
                ProjectionParameterFunctional(parameter_name, len(parameter_basis), i)
            )

        return LincombOperator(operators, coefficients)



    def project_operators(self) -> Dict:
        state_basis = self.bases['state_basis']
        parameter_basis = self.bases['parameter_basis']

        if len(state_basis) == 0:
            state_basis = None

        if len(parameter_basis) == 0:
            parameter_basis = None

        unconstant_operator, constant_operator = split_constant_and_parameterized_operator(
            complete_operator=self._assemble_reduced_operator()
        )

        # At the moment only unique kind of parameter is supported.
        assert len(self.FOM.model_parameter['parameters']) == 1
        projected_parameters = Parameters(
            {list(self.FOM.model_parameter['parameters'].keys())[0] : len(self.bases['parameter_basis'])}
        )
        
        if parameter_basis:
            Q = NumpyVectorSpace(dim = len(parameter_basis))
        else:
            Q = self.FOM.Q

        if state_basis:
            V = NumpyVectorSpace(dim = len(state_basis))
        else:
            V = self.FOM.V

        A = AssembledA(
            unconstant_operator = unconstant_operator,
            constant_operator = constant_operator,
            parameters=projected_parameters,
            Q = Q
        )
        B = AssembledB(
            unconstant_operator = unconstant_operator, 
            constant_operator = constant_operator,
            V = V
        )

        projected_operators = {
            'u_0' : V.make_array(self.project_vectorarray(self.FOM.u_0, basis='state_basis')),
            'M' : project(self.FOM.M, state_basis, state_basis),
            'A' : A,
            'f' : project(self.FOM.f, state_basis, None),
            'B' : B, 
            'constant_cost_term' : self.FOM.constant_cost_term,
            'linear_cost_term' : project(self.FOM.linear_cost_term, state_basis, None),
            'bilinear_cost_term' : project(self.FOM.bilinear_cost_term, state_basis, state_basis),
            'Q' : Q,
            'V' : V,
            'q_circ' : Q.make_array(self.project_vectorarray(self.FOM.q_circ, basis='parameter_basis')),
            'constant_reg_term' : self.FOM.constant_reg_term,
            'linear_reg_term' : project(self.FOM.linear_reg_term, state_basis, None),
            'bilinear_reg_term' : project(self.FOM.bilinear_reg_term, state_basis, state_basis),
            'products' : {
                'prod_H' : project(self.FOM.products['prod_H'], state_basis, state_basis),
                'prod_Q' : project(self.FOM.products['prod_Q'], parameter_basis, parameter_basis),
                'prod_V' : project(self.FOM.products['prod_V'], state_basis, state_basis),
                'prod_C' : self.FOM.products['prod_C'],
                'bochner_prod_Q' : None # TODO Build bochner_prod_Q in discretizer.py as NumpyMatrixOperator 
            }

        }
        return projected_operators 

    def build_rom(self):
        dims = {
            'N': None,
            'nt': self.FOM.dims['nt'],
            'fine_N': None,
            'state_dim': len(self.bases['state_basis']),
            'fine_state_dim': None,
            'diameter': None,
            'fine_diameter': None,
            'par_dim': len(self.bases['parameter_basis']),
            'output_dim': self.FOM.dims['output_dim']                                                                                                                                                                     # options to preassemble affine components or not
        }

        model_parameter = self.FOM.model_parameter.copy()
        model_parameter['q_circ'] = self.project_vectorarray(self.FOM.q_circ, basis='parameter_basis')
        model_parameter['q_exact'] = None
        model_parameter['bounds'] = [self.bases['parameter_basis'].to_numpy().dot(self.FOM.model_parameter['bounds'][0]), 
                                     self.bases['parameter_basis'].to_numpy().dot(self.FOM.model_parameter['bounds'][1]) ]
        
        # At the moment only unique kind of parameter is supported.
        assert len(model_parameter['parameters']) == 1
        model_parameter['parameters'] = Parameters({
            list(self.FOM.model_parameter['parameters'].keys())[0] : len(self.bases['parameter_basis'])
        })
        
        return InstationaryModelIP(
            *(self.project_operators().values()),
            visualizer=self.FOM.visualizer,
            dims = dims,
            model_parameter = model_parameter
        )

    def assemble_error_estimator(self):
        pass

    # TODO write subbasis variants for project_operator and assemble_error_estimator
    def for_subbasis(self):
        pass
