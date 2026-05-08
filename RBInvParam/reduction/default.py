from typing import Dict, List
from timeit import default_timer as timer
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pymor.operators.constructions import LincombOperator, ZeroOperator
from pymor.algorithms.projection import project
from pymor.vectorarrays.interface import VectorArray, VectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional


from RBInvParam.reduction.base import BaseIPReductor
from RBInvParam.reduction.registry import register_reductor
from RBInvParam.products import BochnerProductOperator
from RBInvParam.evaluators import ROMEvaluatorA, EvaluatorLincomb
from RBInvParam.utils.discretization import split_constant_and_parameterized_operator

@register_reductor("default")
class DefaultIPReductor(BaseIPReductor):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cached_operators = {
            "A": None,
            "A_r_state": None,
            "A_r_adjoint": None,
            "A_r_adjoint_state": None,
        }


    def delete_cached_operators(self,
                                targets: List[str] | str | None = 'all') -> None:
        self.logger.debug('Deleting cache')
        assert isinstance(targets, List) or targets == 'all'

        if targets == 'all':
            targets = self._cached_operators.keys()

        assert set(targets).issubset(set(self._cached_operators.keys()))

        for target in targets:
            #del self._cached_operators[target]
            self._cached_operators[target] = None

    def assemble_parameter_reduced_A(self) -> LincombOperator:
        self._logger.debug("Assemble parameter reduced A")
        parameter_basis = self._get_projection_basis('parameter_basis')
        if not parameter_basis:
            parameter_basis = self.FOM.Q.make_array(np.eye(self.FOM.Q.dim))        

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
        if self.offline_parallel:
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
    
    def linearize_A(self,
                    parameter_reduced_A: LincombOperator) -> LincombOperator:
        assert isinstance(parameter_reduced_A, LincombOperator)

        new_operators = []
        new_coefficients = parameter_reduced_A.coefficients

        for op in parameter_reduced_A.operators:

            # If operator is already linear, keep it
            if op.linear:
                new_operators.append(op)

            # Otherwise linearize it
            else:
                # Typical pyMOR linearization call
                lin_op, _ = op.linearize(self.u)   # self.u = current state
                new_operators.append(lin_op)

        return LincombOperator(new_operators, new_coefficients)
        
    def project_A(self, 
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

        # --- offline_parallel / serial path ----------------------------------------------

        self.logger.info("Projecting A(q).")
        t = timer()
        if self.offline_parallel:
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
            A_r = self.project_A(
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
            ),
            'bochner_prod_C' : self.FOM.products['bochner_prod_C']
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