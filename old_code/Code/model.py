# ~~~
# This file is part of the paper:
#
#           " Adaptive Trust Region Reduced Basis Methods for Inverse Parameter Identification Problems "
#
#   https://github.com/michikartmann
#
# Copyright 2023 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Contributors team: Michael Kartmann, Tim Keil
# ~~~
# Description: contains the implementation of the full-order or reduced-order model of the inverse problem.

import numpy as np
import matplotlib.pyplot as plt
import scipy
from helpers import LINSOLVER_with_callback, struct
from pymor.operators.constructions import LincombOperator
from pymor.models.basic import StationaryModel
from pymor.parameters.functionals import ProjectionParameterFunctional, ParameterFunctional
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.discretizers.builtin.grids.referenceelements import square
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace
from scipy.sparse import coo_matrix, csc_matrix
from timeit import default_timer as timer

class StationaryModelIP(StationaryModel):
    def __init__(self, primal_model, primal_data, opt_data, products = None,
                 estimators = None, name = None, hierarchical_model = None,
                 boundary_info = None):

        # TODO 
        # Die 13 Grundbausteine + produkte + notwendige parameter + pyMOR spezifische Sachen initialisieren
        # Grundbausteine kommen aus discretizer oder reductor.
        # A und B sind function bzw. Klassen mit diesen Funktionen um eventuelle Offline ergebnisse zu speichern.
        # Flag um zwischen den Modellen zu unterscheiden. 
        
        # initialize
        self.quadrature_order = 2
        self.__auto_init(locals())
        self.operator = primal_model.operator
        self.rhs = primal_model.rhs
        self.visualizer = primal_model.visualizer
        self.error_estimator = primal_model.error_estimator
        self.parameters = self.opt_data['parameters']
        self.nodes_to_element_projection = self.primal_data['nodes_to_element_projection']
        self.par_dim = self.opt_data['par_dim']
        self.bounds = self.opt_data['bounds']
        if self.products is None:
            self.products = self.primal_model.products   
        self.h1_product = self.products['h1']
        self.l2_product = self.products['l2']
        self.solution_space = primal_model.solution_space
        self.B_u_time = struct()
        self.B_u_time.time_assemb = []
        assert boundary_info is not None 
        
        # check if reaction or diffusion problem and if preassembled or not
        if 'reaction' in self.opt_data['problem_type']:
            self.reaction_problem = True
            self.assembled_model = False
            self.B_assemble = False
        else:
            self.reaction_problem = False
            self.assembled_model = False #self.assemble_diffusion
            self.B_assemble = self.opt_data['B_assemble']
            
        # check which type of model is needed: FOM, FOM with reduced parameter space or ROM
        self.fom = False
        self.Qfom = False
        self.rom = False
        if 'fom' in self.name and not 'Q' in self.name:
            self.fom = True                                                     # fom with infinitedim paramter space
        elif 'fom' in self.name and 'Q' in self.name:
            self.B_assemble = True
            self.assembled_model = True
            self.Qfom = True                                                    # fom with finitedim paramter space
        elif 'rom' in self.name:
            self.B_assemble = True
            self.assembled_model = True
            self.rom = True                                                     #rom with finite dim parameter space 
            
        # check if preassembled or not
        if not self.assembled_model and not self.B_assemble: # unassembled
            assert self.opt_data['par_dim'] ==  self.opt_data['FE_dim'], 'This works only for same meshes for state and control.'
            self.constant_operator = self.operator
            self.prepare_direct_A_q_assembly()
        elif not self.assembled_model and self.B_assemble:
            self.prepare_direct_A_q_assembly()
            self.true_parameterized_operator, self.constant_operator = self.extract_true_parameterized_operator() # tru parametrized wird nur verwendet wenn diffusion
        else: 
            self.true_parameterized_operator, self.constant_operator = self.extract_true_parameterized_operator() # tru parametrized wird nur verwendet wenn diffusion
            
        # TODO 
        # Q_h in disc. oder reductor bauen.
        
        # create Q space
        self.Q_vector_space = NumpyVectorSpace(self.opt_data['par_dim'], 'STATE')
        self.Q_parameter_space = primal_model.parameters.space(self.opt_data['low'], self.opt_data['up'])
        self.parameters = self.primal_model.parameters
        
        # choose Q norm and regularization functional

        # TODO 
        # Q nur P1, if zweig überflüssig
        # Das muss alles nach disc. oder reductor und ist in den Grundbausteinen
        if opt_data['q_in_h1'] is False:                                        # P0 elements for Q
            if opt_data['norm_on_q'] is None or opt_data['norm_on_q'] == 'L2':
               self.opt_data['Q_mass_factor'] = 1/opt_data['par_dim']       
            elif opt_data['norm_on_q'] == 'H1': 
                assert 0, "No H1 Norm available if Q only in L2"
            elif opt_data['norm_on_q'] == 'euclidian': 
                self.opt_data['Q_mass_factor'] = 1      
            self.Q_product = lambda q1, q2: self.opt_data['Q_mass_factor']*q1.T@q2
            self.Q_norm = lambda q: np.sqrt(self.Q_product(q,q))   
            # choose regularization functional
            self.R = lambda q: 0.5*self.Q_product(q,q)
            self.R_d = lambda q: self.opt_data['Q_mass_factor']*q
            self.R_dd = lambda q, d: self.opt_data['Q_mass_factor']*d
        else: # P1 elements for Q
            self.Q_mass_factor = 1
            if self.opt_data['norm_on_q'] is None or self.opt_data['norm_on_q'] == 'H1': # take H1 regularization
                if opt_data['par_dim'] == self.h1_product.matrix.shape[0]:
                    self.prod_matrix = self.h1_product.matrix
                else:
                    self.prod_matrix = self.opt_data['Q_prod_H1_matrix']
                self.Q_prod_py =  NumpyMatrixOperator(self.prod_matrix, source_id = 'STATE', range_id = 'STATE')
                self.Q_product = lambda q1, q2: q1.T@self.prod_matrix.dot(q2)
                self.Q_norm = lambda q: np.sqrt(self.Q_product(q,q))
                self.R = lambda q: 0.5*self.Q_product(q,q)
                self.R_d = lambda q: self.prod_matrix.dot(q)
                self.R_dd = lambda q, d: self.prod_matrix.dot(d)
            elif opt_data['norm_on_q'] == 'L2': 
                if opt_data['par_dim'] == self.h1_product.matrix.shape[0]:
                    self.prod_matrix = self.l2_product.matrix
                else:
                    self.prod_matrix = self.opt_data['Q_prod_L2_matrix']
                self.Q_prod_py = NumpyMatrixOperator(self.prod_matrix, source_id = 'STATE', range_id = 'STATE')
                self.Q_product = lambda q1, q2: q1.T@self.prod_matrix.dot(q2)
                self.Q_norm = lambda q: np.sqrt(self.Q_product(q,q))
                self.R = lambda q: 0.5*self.Q_product(q,q)
                self.R_d = lambda q: self.prod_matrix.dot(q)
                self.R_dd = lambda q, d: self.prod_matrix.dot(d)                
            elif  opt_data['norm_on_q'] == 'euclidian': 
                self.Q_norm_py = None
                #self.prod_matrix = sp.eye(self.opt_data['par_dim'], dtype=np.int8)
                self.Q_product = lambda q1, q2: q1.T@q2
                self.Q_norm = lambda q: np.sqrt(self.Q_product(q,q))
                self.R = lambda q: 0.5*self.Q_product(q,q)
                self.R_d = lambda q: q
                self.R_dd = lambda q, d: d

#%% solve methods

    def clear_dirichlet_dofs(self, rhs):
        if "fom" in self.name:
            if isinstance(rhs, NumpyVectorArray):
                rhs = rhs.to_numpy()[0]
            DI = self.boundary_info.dirichlet_boundaries(2)
            rhs[DI] = 0
        if not isinstance(rhs, NumpyVectorArray):
            new_rhs = self.rhs.range.from_numpy(rhs)
        else:
            new_rhs = rhs
        return new_rhs

    def state(self, q, A_q=None, return_error_estimator=False, solver_option = 'direct'):
        if A_q is None:
            A_q = self.assemble_A_q(q)   
        if solver_option == 'direct':
            u = A_q.apply_inverse(self.rhs.as_range_array())
        else:
            rhs =  self.rhs.as_range_array()
            result, exitCode, num_iter = LINSOLVER_with_callback(A_q.matrix, rhs.to_numpy()[0], 1e-13, 'GMRES')
            self.rhs.as_range_array()
            u = rhs.space.from_numpy(result)
        if return_error_estimator:
           return u,  self.estimate_state(q, u)
        else: 
           return u
     
    def adjoint(self, q, u = None, A_q = None, data_tikonov = 0, solver_option = 'direct'):
        if A_q is None:
            A_q = self.assemble_A_q(q)
        if u is None:
            u = self.state(q, A_q)
        adjoint_rhs = - self.l2_product.apply_adjoint(u+data_tikonov) + self.opt_data["mass_u_noise"]
        if self.boundary_info.has_dirichlet:
            adjoint_rhs = self.clear_dirichlet_dofs(adjoint_rhs)
        if solver_option == 'direct':
            p = A_q.apply_inverse(adjoint_rhs)
        else:
            result, exitCode = scipy.sparse.linalg.gmres(A_q.matrix, adjoint_rhs.to_numpy()[0], tol = 1e-13)
            p = adjoint_rhs.space.from_numpy(result)
        return p
          
    def lin_state(self, q, d, u=None, A_q=None, B_u=None):
        if A_q is None:
            A_q = self.assemble_A_q(q)
        if u is None:
            u = self.state(q, A_q)
        if B_u is None:
            B_u = self.assemble_B_u(u)   
        lin_state_rhs = B_u.B_u(d) 
        if self.boundary_info.has_dirichlet:
            lin_state_rhs = self.clear_dirichlet_dofs(lin_state_rhs)
        u_lin = A_q.apply_inverse(lin_state_rhs)
        return u_lin
       
    def lin_adjoint(self, q , u = None, p = None, u_lin = None,
                    A_q = None , B_u = None, include_rhs = None):
        if include_rhs is not None:
            lin_adjoint_rhs = - self.l2_product.apply_adjoint(u_lin+u) \
                               + self.opt_data["mass_u_noise"]
        else: 
            lin_adjoint_rhs = - self.l2_product.apply_adjoint(u_lin)
        if self.boundary_info.has_dirichlet:
            lin_adjoint_rhs = self.clear_dirichlet_dofs(lin_adjoint_rhs)
        p_lin = A_q.apply_inverse(lin_adjoint_rhs)
        return p_lin
    
#%% assembly methods

    def prepare_direct_A_q_assembly(self):
        g = self.primal_data['grid']
        q, w = g.reference_element.quadrature(order=self.quadrature_order)
        self.quad_points = q
        self.quad_weights = w
        SF_GRAD = self.opt_data['LagrangeShapeFunctionsGrads'][1]
        SF_GRAD = SF_GRAD(q)
        self.SF_GRADS = np.einsum('eij,pjc->epic', g.jacobian_inverse_transposed(0), SF_GRAD)
        self.SF_I0 = np.repeat(g.subentities(0, g.dim), 4, axis=1).ravel()
        self.SF_I1 = np.tile(g.subentities(0, g.dim), [1, 4]).ravel() 
        SF = self.opt_data['LagrangeShapeFunctions'][1]
        self.SF = np.array(tuple(f(q) for f in SF))    
       
    def assemble_A_q_classic(self, mu=None, dirichlet_clear = True):
        g = self.primal_data['grid']
        bi = self.primal_data['boundary_info']
        if not self.reaction_problem:
            _, w = g.reference_element.quadrature(order=self.quadrature_order)
            SF_GRADS = self.SF_GRADS
            SF_I0 = self.SF_I0
            SF_I1 = self.SF_I1
            D = self.nodes_to_element_projection.dot(mu)
            SF_INTS = np.einsum('epic,eqic,c,e,e->epq', SF_GRADS, SF_GRADS, w, g.integration_elements(0), D).ravel()
            del D 
            if bi.has_dirichlet and dirichlet_clear:
                SF_INTS = np.where(bi.dirichlet_mask(g.dim)[SF_I0], 0, SF_INTS) 
            A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
            del SF_INTS, SF_I0, SF_I1
            A.eliminate_zeros()
            A = csc_matrix(A).copy()
        else:
            g = self.primal_data['grid']
            bi = self.primal_data['boundary_info']
            _, w = square.quadrature(order=self.quadrature_order)
            C = self.nodes_to_element_projection.dot(mu)
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
    
    # TODO Landet in der Klasse, die A händelt
    def extract_true_parameterized_operator(self):
        operators, coefficients = [], []
        constant_operators, constant_coefficients = [], []
        for coef, op in zip(self.operator.coefficients, self.operator.operators):
            assert not op.parametric, 'B operator needs to be a true LincombOperator'
            if isinstance(coef, ParameterFunctional) and coef.parametric:
                # then the operator is parametric
                assert isinstance(coef, ProjectionParameterFunctional), 'other cases are not implemented yet'
                operators.append(op)
                coefficients.append(coef)
            else:
                constant_operators.append(op)
                constant_coefficients.append(coef)
        constant_operator = LincombOperator(constant_operators, constant_coefficients).assemble()
        true_parameterized_operator = LincombOperator(operators, coefficients, name='true_parameterized_operator')
        return true_parameterized_operator, constant_operator
 
    def assemble_A_q(self, q):                                                  # asembly of stiffness matrix for (state, adjoint, lin state, lin adjoint)
        if not self.assembled_model:
            A_q = self.assemble_A_q_classic( q)
            A_q = (NumpyMatrixOperator(A_q, source_id = 'STATE', range_id = 'STATE') + self.constant_operator).assemble()            
        else:
            q_as_par = self.parameters.parse(q)
            A_q = self.operator.assemble(q_as_par)
        return A_q

    def B_u_unassembled_reaction(self, u, A_u, v):
        if isinstance(v, NumpyVectorArray):
            v = v.to_numpy().reshape((self.opt_data['FE_dim'],))
        elif isinstance(v,np.ndarray):
            pass
        else:
            assert 1, 'wrong input here...'
        return -u.space.from_numpy(A_u.dot(v))
    
    def assemble_B_u_advection(self, u):
        g = self.primal_data['grid']
        U = u[g.subentities(0, g.dim)]
        quad_, _ = g.reference_element.quadrature(order=1)
        SF_ = self.opt_data['LagrangeShapeFunctions'][1]
        SF_ = np.array(tuple(f(quad_) for f in SF_)).reshape((4,))
        SF_INTS = np.einsum('p,eqic,esic,es,c,e->eqp', SF_, self.SF_GRADS, self.SF_GRADS, U, self.quad_weights, g.integration_elements(0)).ravel()
        SF_I0 = np.repeat(g.subentities(0, g.dim), 4, axis=1).ravel()
        SF_I1 = np.tile(g.subentities(0, g.dim), [1, 4]).ravel() 
        out = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim), g.size(g.dim)))
        out = csc_matrix(out).copy()
        return -out
    
    def assemble_B_u(self, u):                                                  # return linearization B(u) = -partial_q a(u,v) at current state u
        start_time = timer()
        B_u = struct()
        if not self.B_assemble:
            if self.reaction_problem:
                g = self.primal_data['grid']
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
        
        end_time = timer() - start_time
        self.B_u_time.time_assemb.append(end_time)
        return B_u
        
#%% methods for IRGNM tikonov step

    def solve_and_assemble(self, q, solver_option = 'direct'):                 # method returns state, adjoint and diffusionmatrix at parameter q, B_u
        A_q = self.assemble_A_q(q)                                             # assemble A_q global stiffness matrix
        u = self.state(q, A_q, solver_option = solver_option)
        p = self.adjoint(q, u, A_q, solver_option = solver_option)
        return u, p, A_q
    
    def solve_and_assemble_state(self,q, solver_option = 'direct'):
        A_q = self.assemble_A_q(q)                                             # assemble A_q global stiffness matrix
        u = self.state(q,A_q, solver_option = solver_option)
        return u, A_q

    def prepare_nullspace_CG(self, A_q, B_u, p, q, alpha):
        B_u_T_p = B_u.B_u_ad(p, 'grad')
        cg_rhs = self.R_d(alpha*(self.opt_data["q_0"] - q)) + B_u_T_p          # build cg rhs
        cg_lhs = lambda d: self.nullspace_CG_lhs(A_q,B_u,alpha,d)              # build cg lhs
        return cg_rhs, cg_lhs, B_u_T_p
    
    def get_nullspace_CG_lhs(self, A_q, B_u, alpha):
        cg_lhs = lambda d: self.nullspace_CG_lhs(A_q,B_u,alpha,d)              # build cg lhs
        return cg_lhs
 
    def nullspace_CG_lhs(self, A_q, B_u, alpha, d):
        
        u_lin = self.lin_state(q= None, d = d, u=None, A_q = A_q, B_u= B_u)
        p_lin_partial = self.lin_adjoint(q = None , u = None, p = None, u_lin = u_lin,
                        A_q = A_q , B_u = B_u, include_rhs = None)
        B_u_T_p_lin = B_u.B_u_ad(p_lin_partial, 'cg') 
        return self.R_d(alpha*d) - B_u_T_p_lin, u_lin, p_lin_partial

    def gradient(self, B_u, p, q, tikonov_parameter = None):
        B_u_T_p = B_u.B_u_ad(p, 'grad') 
        if tikonov_parameter is not None:
           return tikonov_parameter*self.R_d(q-self.opt_data["q_0"]) - B_u_T_p
        else: 
           return -B_u_T_p

# projected newton methods
    def linearized_objective(self, q,d=None, u_q=None, u_q_d= None, tikonov_parameter = 0):
        if u_q is None:
            pass
        if u_q_d is None:
            pass
        if tikonov_parameter != 0:
            tikonov_term = tikonov_parameter * self.R(q-self.opt_data["q_0"])
        else: 
             tikonov_term = 0
        return 0.5*self.lin_output_res(u_q, u_q_d, q, d)**2 + tikonov_term
               
    def linearized_objective_gradient(self, q, d, u, A_q, B_u, tikonov_parameter = 0):
        u_lin = self.lin_state(q= None, d = d, u=None, A_q = A_q, B_u= B_u)
        p_lin = self.lin_adjoint(q = None , u = u, p = None, u_lin = u_lin,
                        A_q = A_q , B_u = B_u, include_rhs = 1)  
        B_u_T_p_lin = B_u.B_u_ad(p_lin, 'cg')    
        return self.R_d*(tikonov_parameter*(q+d-self.opt_data["q_0"]))-B_u_T_p_lin
        
    def linearized_objective_hessian(self, dd, A_q, B_u, tikonov_parameter = 0):  
        u_lin = self.lin_state(q= None, d = dd, u=None, A_q = A_q, B_u= B_u)
        p_lin_partial = self.lin_adjoint(q = None , u = None, p = None, u_lin = u_lin,
                        A_q = A_q , B_u = B_u, include_rhs = None)
        B_u_T_p_lin = B_u.B_u_ad(p_lin_partial, 'cg') 
        return self.R_dd(None, tikonov_parameter*dd) - B_u_T_p_lin
  
#%% output functionals

    # nonlinear output residual
    def nonlin_output_res(self, u = None, q = None):
        if u is None:
           u, _ = self.solve_and_assemble(q)
        bilinear_part = self.l2_product.apply2(u, u)[0,0]
        linear_part = -2 * u.inner(self.opt_data["mass_u_noise"],
                                   product = None)[0,0]
        constant_part = self.opt_data["output_constant_part"]
        non_lin_output_res = np.sqrt(abs(bilinear_part + linear_part +
                                         constant_part))
        return non_lin_output_res

    # linearized output residual
    def lin_output_res(self, u_q, u_q_d, q, d):
        us = u_q_d+u_q
        bilinear_part = self.l2_product.apply2(us,us)[0,0]
        linear_part = -2*us.inner(self.opt_data["mass_u_noise"],
                                  product = None)[0,0]
        constant_part = self.opt_data["output_constant_part"]
        lin_output_res = np.sqrt(abs(bilinear_part + linear_part +
                                         constant_part))
        return lin_output_res
     
    def difference_to_data(self, u):
        return u-self.opt_data["u_noise"]
             
#%% estimators
    # TODO
    # Wird in eine eigene Klasse ausgelagert
    def estimate_state(self, q, u_r = None, A_q_r = None):      
        if u_r is None:
            if A_q_r is None:
               A_q_r = self.assemble_A_q(q)
            u_r = self.state(q, A_q_r)   
        if self.estimators is not None:
            primal_estimator = self.estimators['primal']
            if primal_estimator is not None:
                q_par = self.primal_model.parameters.parse(q)
                return float(primal_estimator.estimate_error(u_r, q_par, self.primal_model))
        else:
            raise NotImplementedError('Model has no primal estimator.')

    def dual_residual(self, u_r, p_r, q):
        if self.estimators is not None:
            dual_estimator = self.estimators['dual_residual']
            if dual_estimator is not None:
                return float(dual_estimator.estimate_error(u_r, p_r, q))
        else:
            raise NotImplementedError('Model has no dual estimator.')
            
    def estimate_dual(self, u_r, p_r, q):
        dual_residual = self.dual_residual(u_r, p_r, q)
        estimator_state = self.estimate_state(q,u_r)
        coercivity_q = self.opt_data['coercivity_q'](self.parameters.parse(q))
        return float((dual_residual + 2*estimator_state)/coercivity_q)
      
    def estimate_output(self,u_r,p_r,q):
        est_state = self.estimate_state(q, u_r)
        dual_residual = self.dual_residual(u_r, p_r, q)
        norm_observation_operator = 1 
        return float(norm_observation_operator**2*est_state**2+dual_residual*est_state)
    
    def estimate_linearized_state(self,B_u_r, u_lin_r, d,q):
        if self.estimators is not None:
            linearized_state_estimator = self.estimators['linearized_state']
            if linearized_state_estimator is not None:
                return linearized_state_estimator.estimate_error(B_u_r, u_lin_r, d,q)
        else:
            raise NotImplementedError('Model has no linearized state estimator.')
            
    def estimate_linearized_dual_residual(self,p_lin_r, u_lin_r, u_r, d, q):
        if self.estimators is not None:
            linearized_dual_estimator = self.estimators['linearized_dual_residual']
            if linearized_dual_estimator is not None:
                return linearized_dual_estimator.estimate_error(p_lin_r, u_lin_r, u_r, d,q)
        else:
            raise NotImplementedError('Model has no linearized dual estimator.')
    
#%% other stuff
    
    def plot_matplotlib(self, q, title = None, save = False, save_title = None, path = None):
        q = self.nodes_to_element_projection.dot(q)
        n = int(np.sqrt(q.shape[0]))
        q_cube = q.reshape((n,n))
        plt.imshow(q_cube, origin='lower')
        plt.xticks(ticks = [], labels = [])
        plt.yticks(ticks = [], labels = [])
        plt.colorbar()
        plt.title(title)
        if save:
            if path is not None:
                save_title = path + save_title
            plt.savefig(save_title, bbox_inches='tight')
        plt.show()
    
    def plot_subplot_para(self, q1, q2, q3, q4, title = None, save = False, save_title = None, path = None):
        q1 = self.nodes_to_element_projection.dot(q1)
        n = int(np.sqrt(q1.shape[0]))
        q_cube1 = q1.reshape((n,n))
        
        q2 = self.nodes_to_element_projection.dot(q2)
        n = int(np.sqrt(q2.shape[0]))
        q_cube2 = q2.reshape((n,n))
        
        q3 = self.nodes_to_element_projection.dot(q3)
        n = int(np.sqrt(q1.shape[0]))
        q_cube3 = q3.reshape((n,n))
        
        q4 = self.nodes_to_element_projection.dot(q4)
        n = int(np.sqrt(q4.shape[0]))
        q_cube4 = q4.reshape((n,n))
        
        q_list = [q_cube1,q_cube2,q_cube3,q_cube4]
        max_q = max(max(q1),max(q2),max(q3),max(q4))
        min_q = min(min(q1),min(q2),min(q3),min(q4))
        
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)#,figsize=(10, 10))
        cmap=cm.get_cmap('viridis')
        normalizer=Normalize(min_q,max_q)
        im=cm.ScalarMappable(norm=normalizer, cmap=cmap)
        title_strings = [r'$q^e$', r'$q^{FOM}$', r'$q^{Qr}$', r'$q^{Qr-V_r}$']
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.3,
                            hspace=0.3)
        count = 0
        for i,ax in enumerate(axes.flat):
            #ax.imshow(q_list[count], origin='lower')
            ax.imshow(q_list[count], origin='lower',cmap=cmap,norm=normalizer)
            ax.set_title(title_strings[i], size=10)
            ax.set_axis_off()
            # ax.xticks(ticks = [], labels = [])
            # ax.yticks(ticks = [], labels = [])
            count += 1
        fig.colorbar(im, ax=axes.ravel().tolist())
        if save:
            if path is not None:
                save_title = path + save_title
            plt.savefig(save_title, bbox_inches='tight')
        #plt.xlim(0, 1)
        plt.show()
        
    def check_parameter_bounds(self,q):
        if not (np.all(q-self.bounds[1] <= 0) and np.all(self.bounds[0]-q <= 0)):
            pass
