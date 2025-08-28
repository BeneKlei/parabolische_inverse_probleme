from typing import Dict, Tuple, Union
import numpy as np
import logging
import scipy
import copy
import pymor_dealii_bindings as pd2


from pymor.reductors.basic import ProjectionBasedReductor
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.vectorarrays.interface import VectorArray, VectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.parameters.base import Parameters
from pymor.tools.floatcmp import float_cmp_all
from pymor.operators.constructions import InverseOperator

from RBInvParam.model import InstationaryModelIP
from RBInvParam.evaluators import ROMEvaluatorA, ROMEvaluatorB
from RBInvParam.utils.discretization import split_constant_and_parameterized_operator
from RBInvParam.products import BochnerProductOperator
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.residuals import StateResidualOperator, AdjointResidualOperator
from RBInvParam.error_estimator import StateErrorEstimator, \
    AdjointErrorEstimator, ObjectiveErrorEstimator

from RBInvParam.problems.elasticity.pymor_dealii_bindings.operator import DealIIMatrixOperator
from RBInvParam.problems.elasticity.pymor_dealii_bindings.vectorarray import DealIIVectorSpace

class InstationaryModelIPReductor(ProjectionBasedReductor):
    def __init__(self, 
                 FOM: InstationaryModelIP, 
                 check_orthonormality: bool =False, 
                 check_tol: float = 1e-3,
                 residual_image_basis_mode: str = 'none',
                 logger: logging.Logger = None):
        
        assert isinstance(FOM, InstationaryModelIP)
        assert 'prod_V' in FOM.products.keys()
        assert 'prod_Q' in FOM.products.keys()

        logging.basicConfig()
        if logger:
            self._logger = logger
        else:
            self._logger = get_default_logger(logger_name=self.__class__.__name__)
            self._logger.setLevel(logging.DEBUG)
        self.logger.debug(f"Setting up {self.__class__.__name__}")

        bases = {
            'state_basis' : FOM.V.empty(),
            'parameter_basis' : FOM.Q.empty()
        }

        products = {
            'state_basis' : FOM.products['prod_V'],
            'parameter_basis' : FOM.products['prod_Q']
        }

        self._cached_operators = {
            'A' : None
        }

        self.FOM = FOM
        super().__init__(FOM, 
                         bases, 
                         products,
                         check_orthonormality=check_orthonormality, 
                         check_tol=check_tol)

        assert residual_image_basis_mode in ['none']
        self.residual_image_basis_mode = residual_image_basis_mode 
        self.logger.debug(f"Using residual image basis mode: '{residual_image_basis_mode}'.")
    
    def delete_cached_operators(self) -> None:
        self.logger.debug('Deleting cache')

        del self._cached_operators['A']
        
        self._cached_operators = {
            'A' : None,
        }
    
    def calc_projection_error(self,
                              x: VectorArray,
                              basis: str,
                              normalize: bool = False) -> float:   
                              
        
        assert isinstance(x, VectorArray)
        assert basis in ['state_basis', 'parameter_basis']
        _basis = self.bases[basis]
    
        if normalize:
            norms = x.norm(self.products[basis])
            x.scal(1/norms)
                    
        if len(_basis) > 0:
            projected_x = self.bases[basis].lincomb(
                self.project_vectorarray(x, basis=basis)
            )
            x.axpy(-1,projected_x)
        
        return np.sqrt(np.sum(self.products[basis].pairwise_apply2(x,x)))
   
    def project_vectorarray(self, 
                            x : VectorArray,
                            basis: str) -> np.ndarray:
        
        assert isinstance(x, VectorArray)
        assert basis in ['state_basis', 'parameter_basis']
        _basis = self.bases[basis]
        
        if len(_basis) == 0:
            return x.to_numpy()
        else:
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
        parameter_basis = self._get_projection_basis('parameter_basis')
                
        #if not self._cached_operators['A']:
        start = 0
        constant_operator = self.FOM.A.get_constant_operator()
        if constant_operator:
            operators = [constant_operator]
            coefficients = [1]
        else:
            operators = []
            coefficients = []
        # else:
        #     operators = list(self._cached_operators['A'].operators)
        #     start = len(operators) - 1
        #     coefficients = [1]
             
        for i in range(start, len(parameter_basis)):
            q_i = parameter_basis[i]
            # TODO Refactor here
            m = pd2.SparseMatrix()
            m.reinit(self.FOM.A(q_i).matrix.get_sparsity_pattern())
            m.copy_from(self.FOM.A(q_i).matrix)
            A_q = DealIIMatrixOperator(
                matrix = m
            )
            operators.append(A_q)
        
        for i in range(len(parameter_basis)):
            coefficients.append(
                ProjectionParameterFunctional(
                    'reduced_parameter', 
                    len(parameter_basis), i
                )
            )
        
        self._cached_operators['A'] = LincombOperator(operators, coefficients)
        
        return self._cached_operators['A']

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
            'nt': self.FOM.nt,
            'state_dim': state_dim,
            'par_dim': par_dim,
            'observation_space_dim': self.FOM.setup['dims']['observation_space_dim']                                                                                                                                                                     # options to preassemble affine components or not
        }

        setup = self.FOM.setup.copy()
        setup['dims'] = dims
        setup['q_circ'] = self.project_vectorarray(self.FOM.q_circ, basis='parameter_basis')
        setup['q_exact'] = None
        setup['bounds'] = None
        
        # # TODO Check how A(q) can be calc without parameter
        # # At the moment only unique kind of parameter is supported.
        # assert len(self.FOM.setup['model_parameter']['parameters']) == 1
        # projected_parameters = Parameters(
        #     {list(self.FOM.setup['model_parameter']['parameters'].keys())[0] : \
        #      len(self.bases['parameter_basis'])}
        # )
        # model_parameter['parameters'] = projected_parameters

        # problem_parameter = self.FOM.setup['problem_parameter'].copy()
        # problem_parameter['N'] = None

        return setup

    def project_operators(self,
                          assembled_parameter_reduced_A: LincombOperator,
                          Q : VectorSpace,
                          V : VectorSpace,
                          setup: Dict) -> Dict:
        
        assert isinstance(assembled_parameter_reduced_A, LincombOperator)
    
        state_basis = self._get_projection_basis('state_basis')
        parameter_basis = self._get_projection_basis('parameter_basis')

        
        reduced_operator = project(assembled_parameter_reduced_A,
                                   state_basis,
                                   state_basis)
        

        parameteric_operator, constant_operator = split_constant_and_parameterized_operator(
            complete_operator=reduced_operator
        )

        A = ROMEvaluatorA(
            source = V,
            range = V,
            Q = Q,
            parameteric_operator = parameteric_operator,
            constant_operator = constant_operator
        )

        B = ROMEvaluatorB(
            source = Q,
            range = V,
            Q = Q,
            V = V,
            parameteric_operator = parameteric_operator,
            constant_operator = constant_operator
        )

        if state_basis:
            if isinstance(self.FOM.L, VectorArray):
                L = V.make_array(
                    #L.inner(self.bases['state_basis'])
                    self.FOM.L.inner(self.bases['state_basis'])
                )
            else:
                L = project(self.FOM.L, state_basis, None)
        else:
            L = self.FOM.L

        #print(self.project_vectorarray(self.FOM.L, basis='state_basis'))
        # print(self.FOM.L.inner(self.bases['state_basis']))
        # print(L)
        # import sys
        # sys.exit()

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

        if len(self.bases['state_basis']) > 0:
            projected_initial_data = {
                key: {
                    order: V.make_array(
                        #self.project_vectorarray(val, basis='state_basis')
                        val.inner(self.bases['state_basis'])
                    )
                    for order, val in subdict.items()
                }
                for key, subdict in self.FOM.initial_data.items()
            }
            #linear_cost_term = self.project_vectorarray(self.FOM.linear_cost_term, basis='state_basis')
            linear_cost_term = self.FOM.linear_cost_term.inner(self.bases['state_basis'])
            linear_cost_term = V.make_array(linear_cost_term)
        else:
            projected_initial_data = self.FOM.initial_data
            linear_cost_term = self.FOM.linear_cost_term



        # m = pd2.SparseMatrix()
        # m.reinit(self.FOM.M.matrix.get_sparsity_pattern())
        # m.copy_from(self.FOM.M.matrix)
        # M = DealIIMatrixOperator(
        #     matrix = m
        # )

        projected_operators = {
            'initial_data' : projected_initial_data,
            'M' : project(self.FOM.M, state_basis, state_basis),
            'A' : A,
            'L' : L,
            'B' : B, 
            'constant_cost_term' : self.FOM.constant_cost_term,
            'linear_cost_term' : linear_cost_term,
            'bilinear_cost_term' : project(self.FOM.bilinear_cost_term, state_basis, state_basis),
            'q_circ' : Q.make_array(self.project_vectorarray(self.FOM.q_circ, basis='parameter_basis')),
            'constant_reg_term' : self.FOM.constant_reg_term,
            'linear_reg_term' : project(self.FOM.linear_reg_term, parameter_basis, None),
            'bilinear_reg_term' : project(self.FOM.bilinear_reg_term, parameter_basis, parameter_basis),
            'products' : products,
            'visualizer' : self.FOM.visualizer,            
            'setup' : setup
        }

        return projected_operators 

    def reduce(self) -> InstationaryModelIP:

        state_basis = self._get_projection_basis('state_basis')
        parameter_basis = self._get_projection_basis('parameter_basis')

        setup = self._build_setup()
        
        if parameter_basis:
            Q = NumpyVectorSpace(dim = len(parameter_basis))
        else:
            Q = self.FOM.Q

        if state_basis:
            V = NumpyVectorSpace(dim = len(state_basis))
        else:
            V = self.FOM.V

        assembled_parameter_reduced_A = self._assemble_parameter_reduced_A()

        model_params = {
            'Q' : Q,
            'V' : V,
            'setup' : setup
        }

        projected_operators = self.project_operators(
            assembled_parameter_reduced_A,
            **model_params
        )
        error_estimators = self.assemble_error_estimator(
            assembled_parameter_reduced_A,
            **model_params
        )
        model_params.update(projected_operators)
        model_params.update(error_estimators)
                  
        return self.build_rom(model_params)

    def build_rom(self, model_params: Dict) -> InstationaryModelIP:
        return InstationaryModelIP(
            **model_params,
        )

    def _estimate_residual_image_basis(self,
                                       basis: str,
                                       mode: str) -> Dict:
        assert basis in ['state_residual_image_basis', 'adjoint_residual_image_basis']
        assert mode in ['none']
        
        ret = {}

        if mode == 'none':
            ret["residual_image_basis"] = None
            ret["A_range"] = self.FOM.V
            #ret["riesz_representative"] = True
            ret["riesz_representative"] = False
            return ret
        else:
            raise ValueError

    def assemble_error_estimator(self,
                                 assembled_parameter_reduced_A: LincombOperator,
                                 Q : VectorSpace,
                                 V : VectorSpace,
                                 setup: Dict) -> Dict:

        assert isinstance(assembled_parameter_reduced_A, LincombOperator)
        state_residual_config = self._estimate_residual_image_basis(
            basis = 'state_residual_image_basis',
            mode = self.residual_image_basis_mode
        )

        adjoint_residual_config = self._estimate_residual_image_basis(
            basis = 'adjoint_residual_image_basis',
            mode = self.residual_image_basis_mode
        )

        # At the moment we allow only that both residuals have the same image basis
        assert state_residual_config == adjoint_residual_config
        residual_config = state_residual_config

        residual_image_basis = residual_config['residual_image_basis']
        A_range = residual_config['A_range']

        if self.residual_image_basis_mode == 'none':
            _Q = self.FOM.Q
            _V = self.FOM.V
            M = self.FOM.M
            A = self.FOM.A
            #
            state_basis = self._get_projection_basis('state_basis')
            bases = self.bases
        else:
            _Q = Q
            _V = V
            state_basis = self._get_projection_basis('state_basis')

            unconstant_operator, constant_operator = split_constant_and_parameterized_operator(
                complete_operator=project(op = assembled_parameter_reduced_A, 
                                        range_basis = residual_image_basis, 
                                        source_basis = state_basis)
            )

            M = project(self.FOM.M, residual_image_basis, state_basis)
            A = ROMEvaluatorA(
                unconstant_operator = unconstant_operator,
                constant_operator = constant_operator,
                source = V,
                range = A_range,
                Q = Q,
                parameters=setup['model_parameter']['parameters']
            )
            bases = {
                'parameter_basis' : self.FOM.Q.empty(),
                'state_basis' : self.FOM.V.empty()
            }

        if residual_image_basis:
            if isinstance(self.FOM.L, VectorArray):
                L = A_range.make_array(
                    self.FOM.L.inner(self.bases['state_basis'])
                )
            else:
                L = project(self.FOM.L, residual_image_basis, None)
        else:
            L = self.FOM.L
            
        projected_state_quantities = {
            'M' : M,
            'A' : A,
            'L' : L,
            'Q' : _Q,
            'V' : _V,
            'riesz_representative' : residual_config['riesz_representative'],
            'products': self.FOM.products,
            'setup' : setup,
            'bases' : bases
        }

        projected_adjoint_quantities = {
            'M' : M,
            'A' : A,
            'linear_cost_term' : project(self.FOM.linear_cost_term, residual_image_basis, None),
            'bilinear_cost_term' : project(self.FOM.bilinear_cost_term, residual_image_basis, state_basis),
            'Q' : _Q,
            'V' : _V,
            'riesz_representative' : residual_config['riesz_representative'],
            'products': self.FOM.products,
            'setup' : setup,
            'bases' : bases
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
            # product = project(self.FOM.products['prod_V'], 
            #                   residual_image_basis, 
            #                   residual_image_basis, 
            #                   product=None)
            product = project(InverseOperator(self.FOM.products['prod_V']), 
                              residual_image_basis, 
                              residual_image_basis, 
                              product=None)
            assert not state_residual_operator.riesz_representative
            assert not adjoint_residual_operator.riesz_representative

        A_coercivity_constant_estimator = self.FOM.model_constants['A_coercivity_constant_estimator']
        A_coercivity_constant_estimator = copy.copy(A_coercivity_constant_estimator)
        A_coercivity_constant_estimator.Q = Q

        model_constants = {
                'A_coercivity_constant_estimator' : A_coercivity_constant_estimator,
                'C_continuity_constant' : self.FOM.model_constants['C_continuity_constant']
        }

        state_error_estimator = StateErrorEstimator(
            state_residual_operator = state_residual_operator,
            A_coercivity_constant_estimator = A_coercivity_constant_estimator,
            Q = Q,
            V = V,
            product = product,
            setup = setup
        )
        adjoint_error_estimator = AdjointErrorEstimator(
            adjoint_residual_operator = adjoint_residual_operator,
            A_coercivity_constant_estimator = A_coercivity_constant_estimator,
            Q = Q,
            V = V,
            product = product,
            setup = setup
        )

        objective_error_estimator = ObjectiveErrorEstimator(
            A_coercivity_constant_estimator = A_coercivity_constant_estimator,
            C_continuity_constant = self.FOM.model_constants['C_continuity_constant']
        )

        # error_estimator = {
        #     'state_error_estimator' : state_error_estimator,
        #     'adjoint_error_estimator' : adjoint_error_estimator,
        #     'objective_error_estimator' : objective_error_estimator,
        #     'model_constants' : model_constants,
        # }

        error_estimator = {
            'state_error_estimator' : None,
            'adjoint_error_estimator' : None,
            'objective_error_estimator' : None,
            'model_constants' : model_constants,
        }

        return error_estimator
