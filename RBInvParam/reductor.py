from typing import Dict, Tuple, Union, List
import numpy as np
import logging
import scipy
import copy
import os, threading, time

import pymor_dealii_bindings as pd2

from concurrent.futures import ThreadPoolExecutor  # use threads, not processes
from concurrent.futures import ProcessPoolExecutor
from timeit import default_timer as timer

from pymor.reductors.basic import ProjectionBasedReductor
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.vectorarrays.interface import VectorArray, VectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.constructions import LincombOperator, ZeroOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.parameters.base import Parameters
from pymor.tools.floatcmp import float_cmp_all
from pymor.operators.constructions import InverseOperator
from pymor.parallel.default import new_parallel_pool

from RBInvParam.model import InstationaryModelIP
from RBInvParam.evaluators import ROMEvaluatorA, EvaluatorLincomb
#, ROMEvaluatorB
from RBInvParam.utils.discretization import split_constant_and_parameterized_operator
from RBInvParam.products import BochnerProductOperator
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.error_estimators.state_error_estimators import create_state_error_estimator, StateErrorEstimatorType
from RBInvParam.error_estimators.adjoint_error_estimators import create_adjoint_error_estimator, AdjointErrorEstimatorType
from RBInvParam.error_estimators.objective_error_estimators import create_objective_error_estimator
from RBInvParam.error_estimators.residuals import StateResidualOperator, AdjointResidualOperator


from RBInvParam.problems.shared.pymor_dealii_bindings.operator import DealIIMatrixOperator
from RBInvParam.problems.shared.pymor_dealii_bindings.vectorarray import DealIIVectorSpace

class InstationaryModelIPReductor(ProjectionBasedReductor):
    def __init__(self,
                 FOM: InstationaryModelIP,
                 error_estimator_types: Dict,
                 check_orthonormality: bool = True,
                 check_tol: float = 1e-9,
                 residual_image_basis_mode: str = 'none',
                 parallel: bool = False,
                 active_bases: List[str] = None,
                 use_adjoint_space: bool = False,
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

        self.active_bases = active_bases
        self.FOM = FOM
        self.parallel = parallel
        self.use_adjoint_space = use_adjoint_space
        if self.use_adjoint_space:
            assert 'adjoint_basis' not in self.active_bases

        if self.parallel:
            self.logger.debug(f"Using parallelizatzion for ROM-projection.")

        bases = {
            'parameter_basis' : FOM.Q.empty(),
            'state_basis' : FOM.V.empty(),
            'adjoint_basis' : FOM.V.empty(),
        }
        assert set(self.active_bases).issubset(bases.keys())

        self.dims_history = {key : [0] for key in bases.keys()}

        products = {
            'parameter_basis' : FOM.products['prod_Q'],
            'state_basis' : FOM.products['prod_V'],
            'adjoint_basis' : FOM.products['prod_V'],
        }

        self._cached_operators = {
            'A' : None,
            'A_r_state' : None,
            'A_r_adjoint' : None,
            'A_r_adjoint_state' : None
        }

        self.pool = new_parallel_pool() if parallel else None
 
        super().__init__(FOM,
                         bases,
                         products,
                         check_orthonormality=check_orthonormality,
                         check_tol=check_tol)

        assert residual_image_basis_mode in ['none']
        self.residual_image_basis_mode = residual_image_basis_mode
        self.error_estimator_types = error_estimator_types
        self.logger.debug(f"Using residual image basis mode: '{residual_image_basis_mode}'.")

    def delete_cached_operators(self,
                                targets: List[str] | str = 'all') -> None:
        self.logger.debug('Deleting cache')
        assert isinstance(targets, List) or targets == 'all'

        if targets == 'all':
            targets = self._cached_operators.keys()

        assert set(targets).issubset(set(self._cached_operators.keys()))

        for target in targets:
            #del self._cached_operators[target]
            self._cached_operators[target] = None

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

    def _assemble_parameter_reduced_A(self) -> LincombOperator:
        self._logger.debug("Assemble parameter reduced A")
        parameter_basis = self._get_projection_basis('parameter_basis')

        if not self._cached_operators['A']:
            start = 0
            translation_operator = self.FOM.A.get_translation_operator()

            if not isinstance(translation_operator, ZeroOperator):
                operators = [translation_operator]
                coefficients = [1]
            else:
                operators = []
                coefficients = []
        else:
            operators = list(self._cached_operators['A'].operators)

            translation_operator = self.FOM.A.get_translation_operator()
            if not isinstance(translation_operator, ZeroOperator):
                coefficients = [1]
                start = len(operators) - 1
            else:
                coefficients = []
                start = len(operators)


        n_ops = len(parameter_basis)
        to_build = range(start, n_ops)

        t = timer()

        self.logger.info("Constructing A(q).")
        if self.parallel:
            max_workers = min(16, n_ops) if n_ops > 0 else 1
            self.logger.info(f"Using ThreadPoolExecutor; max_workers={max_workers}")
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                _params = (parameter_basis[i] for i in to_build)

                new_ops = list(ex.map(
                    self.FOM.A.get_parameteric_operator, 
                    #timed_get_op,
                    _params,
                    chunksize=16
                ))
        else:
            self.logger.info("Running operator construction sequentially.")
            new_ops = [self.FOM.A.get_parameteric_operator(parameter_basis[i]) for i in to_build]

        operators.extend(new_ops)
        print(timer()-t)

        # ---- coefficients are cheap, do them serially ----
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

    def _project_A(self, 
                   parameter_reduced_A: LincombOperator,
                   source_basis: str = 'state_basis',
                   range_basis: str = 'state_basis') -> LincombOperator:
        
        assert isinstance(parameter_reduced_A, LincombOperator)
        if source_basis == range_basis == 'state_basis':
            cache_key = 'A_r_state'
        elif source_basis == range_basis == 'adjoint_basis':
            cache_key = 'A_r_adjoint'
        # elif (source_basis == 'state_basis') and (range_basis == 'adjoint_basis'):
        #     cache_key = 'A_r_adjoint_state'
        elif (source_basis == 'adjoint_basis') and (range_basis == 'state_basis'):
            cache_key = 'A_r_adjoint_state'
        else:
            raise ValueError
        
        self._logger.debug(f"Project A_r onto source_basis = '{source_basis}' and range_basis = '{range_basis}'")

        # --- sizes ---------------------------------------------------------------
        dim_Q_old = self.dims_history['parameter_basis'][-2]
        dim_Q_new = self.dims_history['parameter_basis'][-1]

        dim_source_old = self.dims_history[source_basis][-2]
        dim_source_new = self.dims_history[source_basis][-1]

        dim_range_old = self.dims_history[range_basis][-2]
        dim_range_new = self.dims_history[range_basis][-1]

        # --- bases ---------------------------------------------------------------
        _source_basis = self._get_projection_basis(source_basis)
        _range_basis = self._get_projection_basis(range_basis)

        source_V_old = _source_basis[:dim_source_old]                
        source_W     = _source_basis[dim_source_old:]

        range_V_old = _range_basis[:dim_range_old]
        range_W     = _range_basis[dim_range_old:]

        # --- cached reduced operators --------------------------------------------
        if not self._cached_operators[cache_key]:
            self._cached_operators[cache_key] = project(parameter_reduced_A,
                                                        _range_basis,
                                                        _source_basis)
            return self._cached_operators[cache_key]

        cached_reduced = self._cached_operators[cache_key]
        cached_blocks = [op.matrix for op in cached_reduced.operators]  # list of (dim_V_old, dim_V_old)

        # --- original parameter-reduced operator ---------------------------------
        coefficients = parameter_reduced_A.coefficients
        base_operators = parameter_reduced_A.operators  # list of full-order operators
        n_ops = len(base_operators)
        
        def build_reduced_operator(i: int, 
                                   base_operators : List, 
                                   cached_blocks : List,
                                   source_W : VectorArray, 
                                   source_V_old : VectorArray,
                                   range_V_old : VectorArray, 
                                   range_W : VectorArray) -> np.ndarray:
            A = base_operators[i]           # (n, n)
            if i < dim_Q_old:
                VTAV_old = cached_blocks[i]  # (dim_V_old, dim_V_old)

                AW = A.apply(source_W)
                VTAW = range_V_old.inner(AW)

                if source_basis == range_basis:
                    WTAV = VTAW.T
                else:
                    AV = A.apply(source_V_old)
                    WTAV = range_W.inner(AV)

                WTAW = range_W.inner(AW)

                # assemble
                M = np.empty((dim_range_new, dim_source_new), dtype=np.float64)
                M[:dim_range_old, :dim_source_old] = VTAV_old
                M[:dim_range_old, dim_source_old:] = VTAW
                M[dim_range_old:, :dim_source_old] = WTAV
                M[dim_range_old:, dim_source_old:] = WTAW
            else:
                AV = A.apply(_source_basis)
                M  = _range_basis.inner(AV)

            return M

        # --- parallel / serial path ----------------------------------------------

        self.logger.info("Projecting A(q).")
        t = timer()
        if self.parallel:
            self.logger.info("Projecting operator parallely.") 

            Ms = self.pool.map(
                build_reduced_operator,
                range(n_ops),
                base_operators=self.pool.push(base_operators),
                cached_blocks=self.pool.push(cached_blocks),
                source_W=self.pool.push(source_W),
                source_V_old=self.pool.push(source_V_old),
                range_V_old=self.pool.push(range_V_old),
                range_W=self.pool.push(range_W),
            )

            # self.logger.info(f"Using ThreadPoolExecutor; max_workers={os.cpu_count()}")

            # max_workers = max(1, min(16 or 1, n_ops))
            # with ThreadPoolExecutor(max_workers=max_workers) as ex:
            #     operators = list(ex.map(
            #         build_reduced_operator, 
            #         (i for i in range(n_ops))
            #         #chunksize=4
            #     ))
        else:
            self.logger.info("Projecting operator sequentially.")
            Ms = [build_reduced_operator(
                i,
                base_operators=base_operators,
                cached_blocks=cached_blocks,
                source_W=source_W,
                source_V_old=source_V_old,
                range_V_old=range_V_old,
                range_W=range_W
            ) for i in range(n_ops)]
        
        operators = [NumpyMatrixOperator(matrix=M) for M in Ms]        

        # save new reduced operator
        self._cached_operators[cache_key] = LincombOperator(
            operators=operators,
            coefficients=coefficients
        )
        return self._cached_operators[cache_key]
    
    def project_operators(self,
                          parameter_reduced_A: LincombOperator,
                          Q : VectorSpace,
                          V : VectorSpace,
                          V_ad : VectorSpace,
                          setup: Dict) -> Dict:

        assert isinstance(parameter_reduced_A, LincombOperator)
        if self.use_adjoint_space:
            assert V_ad is not None
            raise NotImplementedError
        
        parameter_basis = self._get_projection_basis('parameter_basis')
        state_basis = self._get_projection_basis('state_basis')
        adjoint_basis = self._get_projection_basis('adjoint_basis')


        if state_basis:
            A_r = self._project_A(
                parameter_reduced_A = parameter_reduced_A,
                source_basis = 'state_basis',
                range_basis = 'state_basis'
            )

            parameteric_operator, translation_operator = split_constant_and_parameterized_operator(
                complete_operator=A_r
            )

            A = ROMEvaluatorA(
                source = V,
                range = V,
                Q = Q,
                parameteric_operator = parameteric_operator,
                translation_operator = translation_operator
            )
        else:
            A = EvaluatorLincomb(
                op = parameter_reduced_A,
                Q = Q,
                A_affine = self.FOM.A.A_affine
            )
        
        if state_basis:
            if isinstance(self.FOM.L, VectorArray):
                L = V.make_array(
                    self.FOM.L.inner(self.bases['state_basis'])
                )
            else:
                L = project(self.FOM.L, state_basis, None)
        else:
            L = self.FOM.L

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

        if self.use_adjoint_space:
            prod_V_ad = project(self.FOM.products['prod_V'], adjoint_basis, adjoint_basis)
            products_ad = {
                'prod_H_ad' : project(self.FOM.products['prod_H'], adjoint_basis, adjoint_basis),
                'prod_V_ad' : prod_V_ad,
                'bochner_prod_V_ad' : BochnerProductOperator(
                    product=prod_V_ad,
                    delta_t=self.FOM.delta_t,
                    space = V_ad,
                    nt = self.FOM.nt
                )
            }
            products.update(products_ad)

        if len(self.bases['state_basis']) > 0:
            linear_cost_term = self.FOM.linear_cost_term.inner(self.bases['state_basis'])
            linear_cost_term = V.make_array(linear_cost_term)
        else:
            linear_cost_term = self.FOM.linear_cost_term

        if self.use_adjoint_space:
            keys_state = ['state', 'lin_state']
            keys_adjoint = ['adjoint', 'lin_adjoint', 'second_adjoint']
        else:
            keys_state = list(self.FOM.initial_data.keys())
            keys_adjoint = []

        projected_initial_data = {}
        for key in keys_state:
            if len(self.bases['state_basis']) > 0:
                projected_initial_data[key] = {}
                subdict = self.FOM.initial_data[key]
            
                for order, val in subdict.items():
                    projected_initial_data[key][order] = V.make_array(
                        val.inner(self.bases['state_basis'])
                    )
            
            else:
                projected_initial_data[key] = self.FOM.initial_data[key]

        for key in keys_adjoint:
            if len(self.bases['adjoint_basis']) > 0:
                projected_initial_data[key] = {}
                subdict = self.FOM.initial_data[key]
            
                for order, val in subdict.items():
                    projected_initial_data[key][order] = V_ad.make_array(
                        val.inner(self.bases['adjoint_basis'])
                    )
            else:
                projected_initial_data[key] = self.FOM.initial_data[key]

        projected_operators = {
            'initial_data' : projected_initial_data,
            'M' : project(self.FOM.M, state_basis, state_basis),
            'A' : A,
            'L' : L,
            'C' : project(self.FOM.C, None, state_basis),
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

        ######################################################################################################

        if not self.use_adjoint_space:
            return projected_operators

        A_ad_r = self._project_A(
            parameter_reduced_A = parameter_reduced_A,
            source_basis = 'adjoint_basis',
            range_basis = 'adjoint_basis'
        )
        parameteric_operator, translation_operator = split_constant_and_parameterized_operator(
            complete_operator=A_ad_r
        )

        A_ad = ROMEvaluatorA(
            source = V_ad,
            range = V_ad,
            Q = Q,
            parameteric_operator = parameteric_operator,
            translation_operator = translation_operator
        )

        A_ad_source_r = self._project_A(
            parameter_reduced_A = parameter_reduced_A,
            source_basis = 'adjoint_basis',
            range_basis = 'state_basis'
        )
        parameteric_operator, translation_operator = split_constant_and_parameterized_operator(
            complete_operator=A_ad_source_r
        )

        # B_ad = ROMEvaluatorB(
        #     source = Q,
        #     range = V_ad,
        #     Q = Q,
        #     V = V,
        #     parameteric_operator = parameteric_operator,
        #     translation_operator = translation_operator
        # )


        if len(self.bases['adjoint_basis']) > 0:
            linear_cost_term_ad = self.FOM.linear_cost_term.inner(self.bases['adjoint_basis'])
            linear_cost_term_ad = V_ad.make_array(linear_cost_term_ad)
        else:
            linear_cost_term_ad = self.FOM.linear_cost_term

        projected_ad_operators = {
            'M_ad' : project(self.FOM.M, adjoint_basis, adjoint_basis),
            'A_ad' : A_ad,
            'linear_cost_term_ad' : linear_cost_term_ad,
            'bilinear_cost_term_ad' : project(self.FOM.bilinear_cost_term, adjoint_basis, state_basis),
        }

        projected_operators.update(projected_ad_operators)
        return projected_operators

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
        print(self.parallel)
        parameter_reduced_A = self._assemble_parameter_reduced_A()
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
