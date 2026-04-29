from typing import Dict, Union, List, Optional
from abc import abstractmethod

import numpy as np
import logging
import scipy
import copy


from concurrent.futures import ThreadPoolExecutor  # use threads, not processes
from timeit import default_timer as timer

from pymor.reductors.basic import ProjectionBasedReductor
from pymor.algorithms.projection import project
from pymor.vectorarrays.interface import VectorArray, VectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.constructions import LincombOperator, ZeroOperator
from pymor.tools.floatcmp import float_cmp_all
from pymor.parallel.default import new_parallel_pool

from RBInvParam.model import InstationaryModelIP
from RBInvParam.evaluators import ROMEvaluatorA
from RBInvParam.utils.discretization import split_constant_and_parameterized_operator
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.error_estimators.state_error_estimators import create_state_error_estimator, StateErrorEstimatorType
from RBInvParam.error_estimators.adjoint_error_estimators import create_adjoint_error_estimator, AdjointErrorEstimatorType
from RBInvParam.error_estimators.objective_error_estimators import create_objective_error_estimator
from RBInvParam.error_estimators.residuals import StateResidualOperator, AdjointResidualOperator

from RBInvParam.schemas.reductor import InstationaryReductorConfig

class BaseIPReductor(ProjectionBasedReductor):
    def __init__(
        self,
        FOM: "InstationaryModelIP",
        active_bases: List[str],
        config: InstationaryReductorConfig,
        logger: Optional[logging.Logger] = None,
    ):
        if not isinstance(FOM, InstationaryModelIP):
            raise TypeError(f"FOM must be InstationaryModelIP, got {type(FOM)}")

        required_products = {"prod_V", "prod_Q"}
        missing = required_products.difference(FOM.products.keys())
        if missing:
            raise KeyError(f"FOM.products missing required keys: {sorted(missing)}")

        logging.basicConfig()
        if logger:
            self._logger = logger
        else:
            self._logger = get_default_logger(logger_name=self.__class__.__name__)
            self._logger.setLevel(logging.DEBUG)
        self.logger.debug(f"Setting up {self.__class__.__name__}")
        
        self.FOM = FOM
        self.active_bases = list(active_bases)

        # validate config relative to active_bases
        config.validate(self.active_bases)
        self.config = config

        if self.config.offline_parallel:
            self.logger.debug("Using parallelization for ROM-projection.")

        bases = {
            "parameter_basis": FOM.Q.empty(),
            "state_basis": FOM.V.empty(),
            "adjoint_basis": FOM.V.empty(),
            "linearization_basis": FOM.V.empty(),
        }

        if not set(self.active_bases).issubset(bases.keys()):
            unknown = set(self.active_bases).difference(bases.keys())
            raise KeyError(f"Unknown active_bases entries: {sorted(unknown)}")

        self.dims_history = {key: [0] for key in bases.keys()}

        products = {
            "parameter_basis": FOM.products["prod_Q"],
            "state_basis": FOM.products["prod_V"],
            "adjoint_basis": FOM.products["prod_V"],
            "linearization_basis": FOM.products["prod_V"],
        }

        # keep direct attributes if you like (but sourced from config)
        self.offline_parallel = self.config.offline_parallel
        self.pool = new_parallel_pool() if self.offline_parallel else None

        self.use_adjoint_space = self.config.use_adjoint_space
        self.linearization_method = self.config.linearization_method

        super().__init__(
            FOM,
            bases,
            products,
            check_orthonormality=self.config.check_orthonormality,
            check_tol=self.config.check_tol,
        )

        self.residual_image_basis_mode = self.config.residual_image_basis_mode
        self.error_estimator_types = {
            "state": self.config.error_estimators.state,
            "adjoint": self.config.error_estimators.adjoint,
            "objective": self.config.error_estimators.objective,
        }

        self.logger.debug(
            "Using residual image basis mode: %r.", self.residual_image_basis_mode
        )

    @abstractmethod
    def delete_cached_operators(self,
                                targets: List[str] | str | None  = None) -> None:
        pass

    def remove_basis_vectors(self,
                             basis : str,
                             idxes_remove : List[str]) -> None:
        
        assert basis in self.bases
        assert np.all(0 <= idxes_remove < self.dims_history[basis])

        all_idxes = np.arange(0, self.dims_history[basis])
        idxes_keep = np.setdiff1d(all_idxes, idxes_remove)

        self.logger.debug(f"Removing vectors from 'basis' = {basis}.")

        if basis == 'parameter_basis':
            cache_keys = self._cached_operators.keys()
            for cache_key in cache_keys:
                assert isinstance(self._cached_operators[cache_key], LincombOperator)

                operators = self._cached_operators[cache_key].operators[idxes_keep]
                coefficients = self._cached_operators[cache_key].coefficients[idxes_keep]

                self._cached_operators[cache_key] = LincombOperator(operators, coefficients)
            
            self.dims_history[basis] = len(idxes_keep)
        else:
            raise NotImplementedError

    def calc_projection_residuum(self,
                                 x: VectorArray,
                                 basis: str,
                                 normalize: bool = False) -> float:


        assert isinstance(x, VectorArray)
        assert basis in self.active_bases
        _basis = self.bases[basis]

        if normalize:
            norms = x.norm(self.products[basis])
            x.scal(1/norms)

        if len(_basis) > 0:
            projected_x = self.bases[basis].lincomb(
                self.project_vectorarray(x, basis=basis)
            )
            x.axpy(-1,projected_x)

        return x

    def calc_projection_error(self,
                              x: VectorArray,
                              basis: str,
                              normalize: bool = False) -> float:


        assert isinstance(x, VectorArray)
        assert basis in self.active_bases
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
        assert basis in self.active_bases
        _basis = self.bases[basis]

        if len(_basis) == 0:
            return x.to_numpy()
        else:
            return x.inner(_basis, self.products[basis])
        
    def reconstruct(self,
                    x: VectorArray,
                    basis: str) -> VectorArray:

        assert isinstance(x, VectorArray)
        _basis = self.bases[basis]

        if len(_basis) == 0:
            assert x in _basis.space
            return x
        else:
            return _basis[:x.dim].lincomb(x.to_numpy())
        
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

    def _build_setup(self) -> Dict:

        if len(self.bases['parameter_basis']) == 0:
            par_dim = self.FOM.setup['dims']['par_dim']
        else:
            par_dim = len(self.bases['parameter_basis'])

        if len(self.bases['state_basis']) == 0:
            state_dim = self.FOM.setup['dims']['state_dim']
        else:
            state_dim = len(self.bases['state_basis'])

        if self.use_adjoint_space:
            if len(self.bases['adjoint_basis']) == 0:
                adjoint_dim = self.FOM.setup['dims']['state_dim']
            else:
                adjoint_dim = len(self.bases['state_basis'])
        else:
            adjoint_dim = 0

        dims = {
            'nt': self.FOM.nt,
            'state_dim': state_dim,
            'adjoint_dim': adjoint_dim,
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
    
    @abstractmethod
    def assemble_parameter_reduced_A(self) -> LincombOperator:
        pass

    @abstractmethod
    def linearize_A(self,
                    parameter_reduced_A: LincombOperator) -> LincombOperator:
        pass
        
    @abstractmethod
    def project_A(self, 
                  parameter_reduced_A: LincombOperator,
                  source_basis: str = 'state_basis',
                  range_basis: str = 'state_basis') -> LincombOperator:
        pass

    @abstractmethod
    def project_operators(self,
                          parameter_reduced_A: LincombOperator,
                          Q : VectorSpace,
                          V : VectorSpace,
                          V_ad : VectorSpace,
                          setup: Dict) -> Dict:
        pass

    def reduce(self) -> InstationaryModelIP:

        parameter_basis = self._get_projection_basis('parameter_basis')
        state_basis = self._get_projection_basis('state_basis')
        adjoint_basis = self._get_projection_basis('adjoint_basis')

        setup = self._build_setup()

        if parameter_basis:
            Q = NumpyVectorSpace(dim = len(parameter_basis))
        else:
            Q = self.FOM.Q

        if state_basis:
            V = NumpyVectorSpace(dim = len(state_basis))
        else:
            V = self.FOM.V

        if self.use_adjoint_space:
            if adjoint_basis:
                V_ad = NumpyVectorSpace(dim = len(adjoint_basis))
            else:
                V_ad = self.FOM.V
        else:
            V_ad = None

        t = timer()
        print(".............................................")
        print(self.offline_parallel)
        parameter_reduced_A = self.assemble_parameter_reduced_A()
        print(timer() - t)

        model_params = {
            'Q' : Q,
            'V' : V,
            'V_ad' : V_ad,
            'setup' : setup,
        }

        projected_operators = self.project_operators(
            parameter_reduced_A,
            **model_params
        )
        error_estimators = self.assemble_error_estimator(
            parameter_reduced_A,
            **model_params
        )
        model_params.update(projected_operators)
        model_params.update(error_estimators)

        return self.build_rom(model_params)

    def build_rom(self, model_params: Dict) -> InstationaryModelIP:
        return InstationaryModelIP(
            **model_params,
            use_adjoint_space = self.use_adjoint_space
        )

    def _estimate_residual_image_basis(self,
                                       basis: str,
                                       mode: str) -> Dict:
        assert basis in ['state', 'adjoint']
        assert mode in ['none']

        ret = {}

        if mode == 'none':
            ret["residual_image_basis"] = None
            ret["A_range"] = self.FOM.V
            ret["riesz_representative"] = True
        else:
            raise ValueError
        
        if self.error_estimator_types[basis] == StateErrorEstimatorType.HYPERBOLIC:
            ret['gram_operator'] = self.FOM.products['prod_H']
        elif self.error_estimator_types[basis] == StateErrorEstimatorType.PARABOLIC:
            ret['gram_operator'] = self.FOM.products['prod_V']
        else:
            ret['gram_operator'] = None

        return ret

    def assemble_error_estimator(self,
                                 A_r: LincombOperator,
                                 Q : VectorSpace,
                                 V : VectorSpace,
                                 V_ad : VectorSpace,
                                 setup: Dict) -> Dict:

        if self.use_adjoint_space:
            return {
                'state_error_estimator' : None,
                'adjoint_error_estimator' : None,
                'objective_error_estimator' : None,
                'model_constants' : None,
            }


        assert isinstance(A_r, LincombOperator)
        if self.use_adjoint_space:
            assert V_ad is not None

        state_residual_config = self._estimate_residual_image_basis(
            basis = 'state',
            mode = self.residual_image_basis_mode
        )

        adjoint_residual_config = self._estimate_residual_image_basis(
            basis = 'adjoint',
            mode = self.residual_image_basis_mode
        )

        # At the moment we allow only that both residuals have the same image basis
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
                complete_operator=project(op = A_r,
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
            'gram_operator' : residual_config['gram_operator'],
            'products': self.FOM.products,
            'setup' : setup,
            'bases' : bases,
            'zeta' : self.FOM.state_time_stepper.zeta
        }

        projected_adjoint_quantities = {
            'M' : M,
            'A' : A,
            'linear_cost_term' : project(self.FOM.linear_cost_term, residual_image_basis, None),
            'bilinear_cost_term' : project(self.FOM.bilinear_cost_term, residual_image_basis, state_basis),
            'Q' : _Q,
            'V' : _V,
            'riesz_representative' : residual_config['riesz_representative'],
            'gram_operator' : residual_config['gram_operator'],
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

        # if orthonormal_basis:
        #     product = None
        # else:
        #     # product = project(self.FOM.products['prod_V'],
        #     #                   residual_image_basis,
        #     #                   residual_image_basis,
        #     #                   product=None)
        #     product = project(InverseOperator(self.FOM.products['prod_V']),
        #                       residual_image_basis,
        #                       residual_image_basis,
        #                       product=None)
        #     assert not state_residual_operator.riesz_representative
        #     assert not adjoint_residual_operator.riesz_representative

        A_coercivity_constant_estimator = self.FOM.model_constants['A_coercivity_constant_estimator']
        A_coercivity_constant_estimator = copy.copy(A_coercivity_constant_estimator)
        A_coercivity_constant_estimator.Q = Q

        model_constants = {
                'A_coercivity_constant_estimator' : A_coercivity_constant_estimator,
                'C_continuity_constant' : self.FOM.model_constants['C_continuity_constant']
        }

        state_error_estimator = create_state_error_estimator(
            estimator_type = self.error_estimator_types['state'],
            products = self.FOM.products,
            state_residual_operator = state_residual_operator,
            A_coercivity_constant_estimator = A_coercivity_constant_estimator,
            Q = Q,
            V = V,
            setup = setup
        )
        adjoint_error_estimator = create_adjoint_error_estimator(
            estimator_type = self.error_estimator_types['adjoint'],
            products = self.FOM.products,
            adjoint_residual_operator = adjoint_residual_operator,
            A_coercivity_constant_estimator = A_coercivity_constant_estimator,
            Q = Q,
            V = V,
            setup = setup
        )

        objective_error_estimator = create_objective_error_estimator(
            estimator_type = self.error_estimator_types['objective'],
            A_coercivity_constant_estimator = A_coercivity_constant_estimator,
            C_continuity_constant = self.FOM.model_constants['C_continuity_constant'],
            setup = setup
        )

        error_estimator = {
            'state_error_estimator' : state_error_estimator,
            'adjoint_error_estimator' : adjoint_error_estimator,
            'objective_error_estimator' : objective_error_estimator,
            'model_constants' : model_constants,
        }


        return error_estimator
