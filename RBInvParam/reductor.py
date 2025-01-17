from typing import Dict, Tuple, Union
import numpy as np
import logging
import scipy
import copy

from pymor.reductors.basic import ProjectionBasedReductor
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.vectorarrays.interface import VectorArray, VectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.parameters.base import Parameters
from pymor.tools.floatcmp import float_cmp_all

from RBInvParam.model import InstationaryModelIP
from RBInvParam.evaluators import AssembledA, AssembledB
from RBInvParam.utils.discretization import split_constant_and_parameterized_operator
from RBInvParam.products import BochnerProductOperator
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.residuals import StateResidualOperator, AdjointResidualOperator
from RBInvParam.error_estimator import StateErrorEstimator, AdjointErrorEstimator, CoercivityConstantEstimator


class InstationaryModelIPReductor(ProjectionBasedReductor):
    def __init__(self, 
                 FOM: InstationaryModelIP, 
                 check_orthonormality: bool =False, 
                 check_tol: float = 1e-3,
                 logger: logging.Logger = None,
                 use_residual_image_basis: bool = False):
        
        assert isinstance(FOM, InstationaryModelIP)
        assert 'prod_V' in FOM.products.keys()
        assert 'prod_Q' in FOM.products.keys()

        # logging.basicConfig()
        # if logger:
        #     self._logger = logger
        # else:
        #     self._logger = get_default_logger(self.__class__.__name__)
        #     self._logger.setLevel(logging.DEBUG)
        #     print(self._logger)
        #     print(self.logger)

        bases = {
            'state_basis' : FOM.V.empty(),
            'parameter_basis' : FOM.Q.empty(),
            'state_residual_image_basis' : FOM.Q.empty(),
            'adjoint_residual_image_basis' : FOM.Q.empty(),
        }

        #TODO Remove this if not derived class
        _products = {
            'state_basis' : FOM.products['prod_V'],
            'parameter_basis' : FOM.products['prod_Q']
        }

        self.FOM = FOM
        super().__init__(FOM, 
                         bases, 
                         _products,
                         check_orthonormality=check_orthonormality, 
                         check_tol=check_tol)

        self.use_residual_image_basis = use_residual_image_basis 
        if self.use_residual_image_basis:
            pass
            #TODO Log here
        else:
            pass
            #TODO Log here
        
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
        
    def get_bases_dim(self, basis: str) -> int:
        assert basis in self.bases.keys()
        _basis = self.bases[basis]

        if len(_basis) == 0:
            if basis == 'parameter_basis':
                return self.FOM.setup['dims']['par_dim']        
            else:
                return self.FOM.setup['dims']['state_dim']
        else:
            return len(_basis)

    def _get_projection_basis(self, basis: str) -> Union[None, VectorArray]:
        assert basis in self.bases.keys()
        _basis = self.bases[basis]

        if len(_basis) == 0:
            if basis == 'parameter_basis':
                raise NotImplementedError
            else:
                return None
        else:
            return _basis
    
    def _assemble_parameter_reduced_A(self) -> LincombOperator:
        # TODO Handle basis extention via offset
        parameter_basis = self._get_projection_basis('parameter_basis')

        assert len(self.FOM.setup['model_parameter']['parameters']) == 1
        parameter_name = list(self.FOM.setup['model_parameter']['parameters'].keys())[0] 

        operators = [self.FOM.A.constant_operator]
        coefficients = [1]

        for i in range(len(parameter_basis)):
            q_i = parameter_basis[i].to_numpy()[0]
            A_q = self.FOM.A._assemble_A_q(q_i)
            A_q = NumpyMatrixOperator(A_q,
                                      source_id = self.FOM.A.source.id, 
                                      range_id = self.FOM.A.range.id)
            operators.append(A_q)
            coefficients.append(
                ProjectionParameterFunctional(parameter_name, len(parameter_basis), i)
            )

        return LincombOperator(operators, coefficients)

    def _build_setup(self) -> Dict:

        if len(self.bases['parameter_basis']) == 0:
            par_dim = self.FOM.setup['dims']['par_dim']
        else:
            par_dim = len(self.bases['parameter_basis'])
        
        if len(self.bases['state_basis']) == 0:
            state_dim = self.FOM.setup['dims']['state_dim']
        else:
            state_dim = len(self.bases['state_basis'])
        
        dims = {
            'N': None,
            'nt': self.FOM.nt,
            'fine_N': None,
            'state_dim': state_dim,
            'fine_state_dim': None,
            'diameter': None,
            'fine_diameter': None,
            'par_dim': par_dim,
            'output_dim': self.FOM.setup['dims']['output_dim']                                                                                                                                                                     # options to preassemble affine components or not
        }


        model_parameter = self.FOM.setup['model_parameter'].copy()
        model_parameter['q_circ'] = self.project_vectorarray(self.FOM.q_circ, basis='parameter_basis')
        model_parameter['q_exact'] = None
        model_parameter['bounds'] = \
        [self.bases['parameter_basis'].to_numpy().dot(self.FOM.setup['model_parameter']['bounds'][0]), 
         self.bases['parameter_basis'].to_numpy().dot(self.FOM.setup['model_parameter']['bounds'][1]) ]
        
        # TODO Check how A(q) can be calc without parameter
        # At the moment only unique kind of parameter is supported.
        assert len(self.FOM.setup['model_parameter']['parameters']) == 1
        projected_parameters = Parameters(
            {list(self.FOM.setup['model_parameter']['parameters'].keys())[0] : \
             len(self.bases['parameter_basis'])}
        )
        model_parameter['parameters'] = projected_parameters

        problem_parameter = self.FOM.setup['problem_parameter'].copy()
        problem_parameter['N'] = None

        return {
            'dims' : dims, 
            'problem_parameter' : problem_parameter, 
            'model_parameter' : model_parameter, 
        }

    def project_operators(self) -> Dict:
        state_basis = self._get_projection_basis('state_basis')
        parameter_basis = self._get_projection_basis('parameter_basis')

        assembled_parameter_reduced_A = self._assemble_parameter_reduced_A()

        unconstant_operator, constant_operator = split_constant_and_parameterized_operator(
            complete_operator=project(assembled_parameter_reduced_A,
                                      state_basis,
                                      state_basis)
        )

        setup = self._build_setup()
        
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
            source = V,
            range = V,
            Q = Q,
            parameters=setup['model_parameter']['parameters']
        )
        B = AssembledB(
            unconstant_operator = unconstant_operator, 
            constant_operator = constant_operator,
            source = Q,
            range = V,
            V = V
        )

        prod_Q = project(self.FOM.products['prod_Q'], parameter_basis, parameter_basis)
        prod_V = project(self.FOM.products['prod_V'], state_basis, state_basis)

        products = {
            'prod_H' : project(self.FOM.products['prod_H'], state_basis, state_basis),
            'prod_Q' : prod_Q,
            'prod_V' : prod_V,
            'prod_C' : self.FOM.products['prod_C'],
            'bochner_prod_Q' : BochnerProductOperator(
                product=prod_Q,
                delta_t=self.FOM.delta_t,
                space = Q,
                nt = self.FOM.nt
            ),
            'bochner_prod_V' : BochnerProductOperator(
                product=prod_V,
                delta_t=self.FOM.delta_t,
                space = V,
                nt = self.FOM.nt
            )
        }

        A_coercivity_constant_estimator = self.FOM.model_constants['A_coercivity_constant_estimator']
        A_coercivity_constant_estimator = copy.copy(A_coercivity_constant_estimator)
        A_coercivity_constant_estimator.Q = Q

        model_constants = {
                'A_coercivity_constant_estimator' : A_coercivity_constant_estimator,
                'C_continuity_constant' : self.FOM.model_constants['C_continuity_constant']
        }

        state_error_estimator, adjoint_error_estimator = \
        self._assemble_error_estimator(assembled_parameter_reduced_A = assembled_parameter_reduced_A,
                                       A_coercivity_constant_estimator = A_coercivity_constant_estimator,
                                       Q = Q,
                                       V = V,
                                       products = products,
                                       setup = setup)

        projected_operators = {
            'u_0' : V.make_array(self.project_vectorarray(self.FOM.u_0, basis='state_basis')),
            'M' : project(self.FOM.M, state_basis, state_basis),
            'A' : A,
            'L' : project(self.FOM.L, state_basis, None),
            'B' : B, 
            'constant_cost_term' : self.FOM.constant_cost_term,
            'linear_cost_term' : project(self.FOM.linear_cost_term, state_basis, None),
            'bilinear_cost_term' : project(self.FOM.bilinear_cost_term, state_basis, state_basis),
            'Q' : Q,
            'V' : V,
            'q_circ' : Q.make_array(self.project_vectorarray(self.FOM.q_circ, basis='parameter_basis')),
            'constant_reg_term' : self.FOM.constant_reg_term,
            'linear_reg_term' : project(self.FOM.linear_reg_term, parameter_basis, None),
            'bilinear_reg_term' : project(self.FOM.bilinear_reg_term, parameter_basis, parameter_basis),
            'state_error_estimator' : state_error_estimator,
            'adjoint_error_estimator' : adjoint_error_estimator,
            'objective_error_estimator' : None,
            'products' : products,
            'visualizer' : self.FOM.visualizer,
            'model_constants' : model_constants,
            'setup' : setup
        }
        return projected_operators 

    def build_rom(self, 
                  projected_operators : Dict, 
                  error_estimator : Dict = None):
                  
        return InstationaryModelIP(
            **projected_operators,
        )

    # def reduce(self) -> InstationaryModelIP:
    #     raise NotImplementedError

    def _estimate_residual_image_basis(self,
                                       basis: str) -> VectorArray:
        assert basis in ['state_residual_image_basis', 'adjoint_residual_image_basis']
        raise NotImplementedError


    def _assemble_error_estimator(self,
                                  assembled_parameter_reduced_A: LincombOperator,
                                  A_coercivity_constant_estimator: CoercivityConstantEstimator, 
                                  Q : VectorSpace,
                                  V : VectorSpace,
                                  products : Dict,
                                  setup: Dict) -> Tuple[StateResidualOperator, AdjointResidualOperator]:

        assert isinstance(assembled_parameter_reduced_A, LincombOperator)
        state_basis = self._get_projection_basis('state_basis')

        if self.use_residual_image_basis:
            self.bases['state_residual_image_basis'] = self._estimate_residual_image_basis(
                basis = 'state_residual_image_basis'
            )

            self.bases['adjoint_residual_image_basis'] = self._estimate_residual_image_basis(
                basis = 'adjoint_residual_image_basis'
            )

        state_residual_image_basis = self._get_projection_basis('state_residual_image_basis')
        adjoint_residual_image_basis = self._get_projection_basis('adjoint_residual_image_basis')
        
        # At the moment we allow only that both residuals have the same image basis
        assert state_residual_image_basis == adjoint_residual_image_basis
        residual_image_basis = state_residual_image_basis
    
        if residual_image_basis is None:
            riesz_representative = True
        else:
            riesz_representative = False            
            raise NotImplementedError
        
        unconstant_operator, constant_operator = split_constant_and_parameterized_operator(
            complete_operator=project(op = assembled_parameter_reduced_A, 
                                      range_basis = residual_image_basis, 
                                      source_basis = state_basis)
        )

        M = project(self.FOM.M, residual_image_basis, state_basis)
        A = AssembledA(
            unconstant_operator = unconstant_operator,
            constant_operator = constant_operator,
            source = V,
            range = V,
            Q = Q,
            parameters=setup['model_parameter']['parameters']
        )

        projected_state_quantities = {
            'M' : M,
            'A' : A,
            'L' : project(self.FOM.L, residual_image_basis, None),
            'Q' : Q,
            'V' : V,
            'riesz_representative' : riesz_representative,
            'products': products,
            'setup' : setup
        }

        projected_adjoint_quantities = {
            'M' : M,
            'A' : A,
            'linear_cost_term' : project(self.FOM.linear_cost_term, residual_image_basis, None),
            'bilinear_cost_term' : project(self.FOM.bilinear_cost_term, residual_image_basis, state_basis),
            'Q' : Q,
            'V' : V,
            'riesz_representative' : riesz_representative,
            'products': products,
            'setup' : setup
        }

        state_residual_operator = StateResidualOperator(**projected_state_quantities)
        adjoint_residual_operator = AdjointResidualOperator(**projected_adjoint_quantities)

        if residual_image_basis:
            orthonormal_basis = float_cmp_all(
                self.FOM.Q.make_array(scipy.sparse.identity(self.FOM.setup['dims']['state_dim'])),
                self.FOM.products['prod_V'].apply2(residual_image_basis, residual_image_basis),
                rtol = 1e-16,
                atol = 1e-16,
            )
        else:
            orthonormal_basis = False
                
        if orthonormal_basis:
            product = None
        else:
            product = project(self.FOM.products['prod_V'], 
                              residual_image_basis, 
                              residual_image_basis, 
                              product=None)

        state_error_estiamtor = StateErrorEstimator(
            state_residual_operator = state_residual_operator,
            A_coercivity_constant_estimator = A_coercivity_constant_estimator,
            Q = Q,
            V = V,
            product = product,
            setup = setup
        )
        adjoint_error_estiamtor = AdjointErrorEstimator(
            adjoint_residual_operator = adjoint_residual_operator,
            A_coercivity_constant_estimator = A_coercivity_constant_estimator,
            Q = Q,
            V = V,
            product = product,
            setup = setup
        )
        
        return state_error_estiamtor, adjoint_error_estiamtor


    # TODO write subbasis variants for project_operator and assemble_error_estimator
    def for_subbasis(self):
        pass
