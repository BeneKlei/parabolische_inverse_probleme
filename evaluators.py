import numpy as np
from typing import Union

from pymor.discretizers.builtin.grids.interfaces import BoundaryInfo, Grid
from pymor.operators.numpy import NumpyMatrixOperator
from scipy.sparse import coo_matrix, csc_matrix
import pymor.vectorarrays as VectorArray
from pymor.discretizers.builtin.grids.referenceelements import square
from pymor.operators.interface import Operator
from pymor.vectorarrays.numpy import NumpyVectorArray

from utils import Struct, build_projection

LAGRANGE_SHAPE_FUNCTIONS = {1: [lambda X: (1 - X[..., 0]) * (1 - X[..., 1]),
                    lambda X: (1 - X[..., 1]) * (X[..., 0]),
                    lambda X:     (X[..., 0]) * (X[..., 1]),
                    lambda X:     (X[..., 1]) * (1 - X[..., 0])]}


LAGRANGE_SHAPE_FUNCTIONS_GRAD = {1: lambda X: np.array(([X[..., 1] - 1., X[..., 0] - 1.], # u links
                                            [1. - X[..., 1], - X[..., 0]], #u rechts
                                            [X[..., 1], X[..., 0]], # o rechts
                                            [-X[..., 1], 1. - X[..., 0]]))}# o links


class Evaluator:
    def __init__(self,
                operator: Operator,
                constant_operator : Operator,
                pre_assemble: bool,
                reaction_problem: bool,
                grid: Grid,
                boundary_info: BoundaryInfo,
                ):

        assert pre_assemble or (grid is not None)
        assert pre_assemble or (boundary_info is not None)
        assert pre_assemble or (constant_operator is not None)
        assert constant_operator is not None

        self.operator = operator
        self.constant_operator = constant_operator
        
        self.pre_assemble = pre_assemble # False if FOM, true else (Q-FOM, ROM)
        self.reaction_problem = reaction_problem
        self.grid = grid
        self.boundary_info = boundary_info
        self.quadrature_order = 2

        if not self.pre_assemble:
            self.nodes_to_element_projection, _, _ = build_projection(self.grid)
            self._prepare()

        self.source = self.constant_operator.source
        self.range = self.constant_operator.range

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

class A_evaluator(Evaluator):
    def __init__(self,
                 operator: Operator,
                 constant_operator : Operator,
                 pre_assemble: bool,
                 reaction_problem: bool,
                 grid: Grid,
                 boundary_info: BoundaryInfo,
                 ):

        super().__init__(
            operator = operator,
            constant_operator = constant_operator,
            pre_assemble = pre_assemble,
            reaction_problem = reaction_problem,
            grid = grid,
            boundary_info = boundary_info
        )
        
    
    def __call__(self, q: VectorArray) -> NumpyMatrixOperator:
        if not self.pre_assemble:
            A_q = self._assemble_A_q(q)
            A_q = NumpyMatrixOperator(
                A_q, 
                source_id = self.source.id, 
                range_id = self.range.id,
            ) 
            A_q += self.constant_operator
            A_q = A_q.assemble()            
        else:
            q_as_par = self.parameters.parse(q)
            A_q = self.operator.assemble(q_as_par)
        return A_q

    def _assemble_A_q(self, q: VectorArray,  dirichlet_clear = True) -> csc_matrix:
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
        
class B_evaluator(Evaluator):
    def __init__(self,
                 operator: Operator,
                 constant_operator : Operator,
                 pre_assemble: bool,
                 reaction_problem: bool,
                 grid: Grid,
                 boundary_info: BoundaryInfo,
                ):
        
        super().__init__(
            operator = operator,
            constant_operator = constant_operator,
            pre_assemble = pre_assemble,
            reaction_problem = reaction_problem,
            grid = grid,
            boundary_info = boundary_info
        )
    
    def B_u_unassembled_reaction(self, 
                                 u, 
                                 A_u, 
                                 v : Union[NumpyVectorArray, np.ndarray]
                                 ):
        if isinstance(v, NumpyVectorArray):
            v = v.to_numpy().reshape((self.opt_data['FE_dim'],))
        elif isinstance(v,np.ndarray):
            pass
        else:
            assert 1, 'wrong input here...'
        return -u.space.from_numpy(A_u.dot(v))
    
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
        B_u = Struct()
        if not self.B_assemble:
            if self.reaction_problem:
                g = self.grid
                _, w = square.quadrature(order=self.quadrature_order)
                C = self.nodes_to_element_projection.dot(u.to_numpy()[0])
                SF_INTS = np.einsum('iq,jq,q,e,e->eij', self.SF, self.SF, w, g.integration_elements(0), C).ravel()
                del C
                SF_I0 = np.repeat(g.subentities(0, g.dim), 4, axis=1).ravel()
                SF_I1 = np.tile(g.subentities(0, g.dim), [1, 4]).ravel()
                A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
                del SF_INTS, SF_I0, SF_I1
                A_u = csc_matrix(A).copy()
                B_u.B_u = lambda d:  self.B_u_unassembled_reaction(u,A_u, d) # numpy -> pymor
                B_u.B_u_ad = lambda p, mode:  self.B_u_unassembled_reaction(u, A_u, p.to_numpy()[0]).to_numpy()[0]  # pmyor -> numpy
            else:
                B_u_mat = self.assemble_B_u_advection(u.to_numpy()[0])
                B_u.B_u = lambda d: u.space.from_numpy(B_u_mat.dot(d))
                B_u.B_u_ad = lambda p, mode: B_u_mat.T.dot(p.to_numpy()[0])
        else:
            DoFs = self.solution_space.dim
            B_u_list = np.zeros((len(self.true_parameterized_operator.operators), DoFs, 1))
            for i, op in enumerate(self.true_parameterized_operator.operators):
                B_u_list[i] = -op.apply_adjoint(u).to_numpy().T
            
            B_u.B_u = lambda d: u.space.from_numpy(np.einsum("tij,t->ij", B_u_list, d).flatten())  # numpy -> pymor
            B_u.B_u_ad = lambda p, mode: np.einsum("tij,i->t", B_u_list, p.to_numpy()[0]) # pymor -> numpy 
        
        return B_u