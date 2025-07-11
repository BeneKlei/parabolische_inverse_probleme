import numpy as np
from typing import Union, Dict

import pymor.vectorarrays as VectorArray
from pymor.discretizers.builtin.grids.interfaces import BoundaryInfo, Grid
from pymor.operators.numpy import NumpyMatrixOperator
from scipy.sparse import coo_matrix, csc_matrix
from pymor.discretizers.builtin.grids.referenceelements import square
from pymor.operators.interface import Operator
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.operators.constructions import LincombOperator
from pymor.parameters.base import Parameters
from pymor.vectorarrays.interface import VectorSpace

from RBInvParam.utils.discretization import Struct, build_projection


LAGRANGE_SHAPE_FUNCTIONS = {1: [lambda X: (1 - X[..., 0]) * (1 - X[..., 1]),
                    lambda X: (1 - X[..., 1]) * (X[..., 0]),
                    lambda X:     (X[..., 0]) * (X[..., 1]),
                    lambda X:     (X[..., 1]) * (1 - X[..., 0])]}


LAGRANGE_SHAPE_FUNCTIONS_GRAD = {1: lambda X: np.array(([X[..., 1] - 1., X[..., 0] - 1.], # u links
                                            [1. - X[..., 1], - X[..., 0]], #u rechts
                                            [X[..., 1], X[..., 0]], # o rechts
                                            [-X[..., 1], 1. - X[..., 0]]))}# o links

# TODO Rename unconstant_operator


# FOM Unassebled
# Q-FOM Assebled
# ROM Assebled
class UnAssembledEvaluator:
    def __init__(self,
                 constant_operator : Operator,
                 reaction_problem: bool,
                 grid: Grid,
                 boundary_info: BoundaryInfo,
                 source : VectorSpace,
                 range : VectorSpace):
        
        assert grid is not None
        assert boundary_info is not None
        #assert reaction_problem 
    
        self.constant_operator = constant_operator
        self.reaction_problem = reaction_problem
        self.grid = grid
        self.boundary_info = boundary_info
        self.quadrature_order = 2

        self.source = source
        self.range = range

        self.nodes_to_element_projection, _, _ = build_projection(self.grid)
        self._prepare()

    def _prepare(self):
        g = self.grid
        q, w = g.reference_element.quadrature(
            order=self.quadrature_order
        )
        self.quad_points = q
        self.quad_weights = w
        SF_GRAD = LAGRANGE_SHAPE_FUNCTIONS_GRAD[1]
        SF_GRAD = SF_GRAD(q)
        self.SF_GRADS = np.einsum(
            'eij,pjc->epic', 
            g.jacobian_inverse_transposed(0), 
            SF_GRAD
        )
        self.SF_I0 = np.repeat(g.subentities(0, g.dim), 4, axis=1).ravel()
        self.SF_I1 = np.tile(g.subentities(0, g.dim), [1, 4]).ravel() 
        SF = LAGRANGE_SHAPE_FUNCTIONS[1]
        self.SF = np.array(tuple(f(q) for f in SF))    

class UnAssembledA(UnAssembledEvaluator):
    def __init__(self,
                 constant_operator : Operator,
                 reaction_problem: bool,
                 grid: Grid,
                 boundary_info: BoundaryInfo,
                 source : VectorSpace,
                 range : VectorSpace,
                 Q : VectorSpace):
        
        super().__init__(
            constant_operator = constant_operator,
            reaction_problem = reaction_problem,
            grid = grid,
            boundary_info = boundary_info,    
            source = source,
            range = range)

        self.Q = Q
        
    
    def __call__(self, q: VectorArray) -> NumpyMatrixOperator:
        assert q in self.Q
        # TODO Check _assemble_A_q can be vectorized
        assert len(q) == 1

        A_q = self._assemble_A_q(q.to_numpy()[0])
        if self.source:
            A_q = NumpyMatrixOperator(
                A_q, 
                source_id = self.source.id, 
                range_id = self.range.id,
            ) 
        else:
            A_q = NumpyMatrixOperator(A_q)

        if self.constant_operator:
            A_q += self.constant_operator

        A_q = A_q.assemble()            
        return A_q

    def _assemble_A_q(self, 
                      q: np.array,  
                      dirichlet_clear: bool = True) -> csc_matrix:
        g = self.grid
        bi = self.boundary_info
        if not self.reaction_problem:
            _, w = g.reference_element.quadrature(order=self.quadrature_order)
            SF_GRADS = self.SF_GRADS
            SF_I0 = self.SF_I0
            SF_I1 = self.SF_I1
            D = self.nodes_to_element_projection.dot(q)
            SF_INTS = np.einsum('epic,eqic,c,e,e->epq', SF_GRADS, SF_GRADS, w, g.integration_elements(0), D).ravel()
            del D 
            if bi.has_dirichlet and dirichlet_clear:
                SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I0], 0, SF_INTS) 
            A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
            del SF_INTS, SF_I0, SF_I1
            A.eliminate_zeros()
            A = csc_matrix(A).copy()
        else:
            g = self.grid
            bi = self.boundary_info
            _, w = square.quadrature(order=self.quadrature_order)
            C = self.nodes_to_element_projection.dot(q)
            SF_INTS = np.einsum('iq,jq,q,e,e->eij', self.SF, self.SF, w, g.integration_elements(0), C).ravel()
            del C
            SF_I0 = np.repeat(g.subentities(0, g.dim), 4, axis=1).ravel()
            SF_I1 = np.tile(g.subentities(0, g.dim), [1, 4]).ravel()
            if bi.has_dirichlet and dirichlet_clear:
               SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I0], 0, SF_INTS)
            A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
            del SF_INTS, SF_I0, SF_I1
            A = csc_matrix(A).copy()

        return A

class AssembledEvaluator():
    def __init__(self,
                 unconstant_operator: Operator,
                 constant_operator : Operator,
                 source : VectorSpace,
                 range : VectorSpace):
        
        assert isinstance(unconstant_operator, LincombOperator)
        assert unconstant_operator or constant_operator
        
        self.unconstant_operator = unconstant_operator
        self.constant_operator = constant_operator

        if self.unconstant_operator and self.constant_operator:
            assert self.unconstant_operator.source == self.constant_operator.source
            assert self.unconstant_operator.range == self.constant_operator.range

        self.source = source
        self.range = range


class AssembledA(AssembledEvaluator):    
    def __init__(self, 
                 unconstant_operator: Operator,
                 constant_operator : Operator,
                 source : VectorSpace,
                 range : VectorSpace,
                 Q : VectorSpace,
                 parameters: Parameters):
        
        super().__init__(unconstant_operator, 
                         constant_operator,
                         source = source,
                         range = range)
        self.Q = Q
        self.parameters = parameters

    def __call__(self, q: VectorArray) -> NumpyMatrixOperator:
        assert q in self.Q
        # TODO Can _assemble_A_q be vectorized?
        assert len(q) == 1

        q_as_par = self.parameters.parse(q.to_numpy()[0])
        return self.unconstant_operator.assemble(q_as_par) + self.constant_operator
    

class UnAssembledB(UnAssembledEvaluator):
    def __init__(self,
                 reaction_problem: bool,
                 grid: Grid,
                 boundary_info: BoundaryInfo,
                 source : VectorSpace,
                 range : VectorSpace,
                 Q : VectorSpace,
                 V : VectorSpace):
        
        super().__init__(
            constant_operator = None,
            reaction_problem = reaction_problem,
            grid = grid,
            boundary_info = boundary_info,
            source=source,
            range=range)

        self.Q = Q
        self.V = V
        assert Q.dim == V.dim # The current implementation only works if Q and V have the same dim
        
    def assemble_B_u_advection(self, u : VectorArray):
        g = self.grid
        U = u[g.subentities(0, g.dim)]
        quad_, _ = g.reference_element.quadrature(order=1)
        SF_ = LAGRANGE_SHAPE_FUNCTIONS[1]
        SF_ = np.array(tuple(f(quad_) for f in SF_)).reshape((4,))
        SF_INTS = np.einsum('p,eqic,esic,es,c,e->eqp', SF_, self.SF_GRADS, self.SF_GRADS, U, self.quad_weights, g.integration_elements(0)).ravel()
        SF_I0 = np.repeat(g.subentities(0, g.dim), 4, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, g.dim), [1, 4]).ravel() 
        out = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
        out = csc_matrix(out).copy()
        return -out
    
    def __call__(self, u: VectorArray) -> NumpyMatrixOperator:
        assert u in self.V
        # TODO Check how this function can be vectorized
        assert len(u) == 1
        u = u.to_numpy()[0]

        B_u = Struct()
        if self.reaction_problem:
            g = self.grid
            _, w = square.quadrature(order=self.quadrature_order)
            C = self.nodes_to_element_projection.dot(u)
            SF_INTS = np.einsum('iq,jq,q,e,e->eij', self.SF, self.SF, w, g.integration_elements(0), C).ravel()
            del C
            SF_I0 = np.repeat(g.subentities(0, g.dim), 4, axis=1).ravel()
            SF_I1 = np.tile(g.subentities(0, g.dim), [1, 4]).ravel()
            A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
            del SF_INTS, SF_I0, SF_I1
            B_u_mat = -csc_matrix(A).copy()
        else:
            B_u_mat = self.assemble_B_u_advection(u)

        B_u.B_u = lambda d: self.V.from_numpy(B_u_mat.dot(d.to_numpy()[0]))
        B_u.B_u_ad = lambda p: self.Q.make_array(B_u_mat.T.dot(p.to_numpy()[0]))
        return B_u
    

class AssembledB(AssembledEvaluator):
    def __init__(self, 
                unconstant_operator: Operator,
                constant_operator : Operator,
                source : VectorSpace,
                range : VectorSpace,
                Q : VectorSpace,
                V : VectorSpace):
    
        super().__init__(unconstant_operator, 
                         constant_operator,
                         source = source,
                         range = range)

        self.Q = Q
        self.V = V

    def __call__(self, u: VectorArray) -> Struct:
        assert u in self.V
        # TODO Check how this function can be vectorized
        assert len(u) == 1

        B_u = Struct()
        if self.unconstant_operator:
            DoFs = u.space.dim
            B_u_list = np.zeros((len(self.unconstant_operator.operators), DoFs))
            for i, op in enumerate(self.unconstant_operator.operators):
                B_u_list[i] = -op.apply_adjoint(u).to_numpy()[0,:]
            
            B_u.B_u = lambda d: self.V.make_array(np.einsum("ti,t->i", B_u_list, d.to_numpy()[0]))
            B_u.B_u_ad = lambda p: self.Q.make_array(np.einsum("ti,i->t", B_u_list, p.to_numpy()[0]))
        else:
            raise NotImplementedError

        return B_u