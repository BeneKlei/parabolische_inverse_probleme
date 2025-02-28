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
# Description: the reductor takes a full-order model and reduces it to create a reduced-order model.
    
from pymor.reductors.coercive import CoerciveRBReductor, SimpleCoerciveRBReductor, SimpleCoerciveRBEstimator
from pymor.operators.constructions import VectorOperator, LincombOperator
from model import StationaryModelIP
from pymor.algorithms.projection import project
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.basic import StationaryRBReductor
from pymor.core.base import ImmutableObject
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.models.basic import StationaryModel
import numpy as np
from pymor.algorithms.gram_schmidt import gram_schmidt
from copy import deepcopy

class CoerciveRBReductorIP(CoerciveRBReductor):
    def __init__(self,
                 Qfom,
                 RB = None,
                 coercivity_estimator = None,
                 product = None,
                 reductor_type = "non_assembled",
                 name = None,
                 RB_big = None,
                 old_data = None):
        self.__auto_init(locals())
        self.fom = Qfom
        self.coarse_options = None
        self.primal_reductor = self.get_primal_reductor()                      
        self.opt_data = self.fom.opt_data.copy()
        self.bases = dict(RB=RB, RB_big = RB_big)
        self.hierarchical_RB = RB_big
        self.primal_reductor_big = None
        
        # get old data from last reductor/model
        if old_data is not None:
            self.old_rom = old_data['old_rom']
            self.old_reductor = old_data['old_reductor']
            if self.old_reductor is not None:
                self.dual_rhs = self.old_reductor.dual_rhs
            else:
                self.dual_rhs = None
            self.old_basis_Q_len = old_data['old_basis_Q_len']
            self.basis_Q_diff = old_data['basis_diff_Q']
        else:
            self.old_rom = None
            self.old_reductor = None
            self.dual_rhs = None
            self.old_basis_Q_len = 0
            self.basis_Q_diff = 0

#%% methods for basis extending/coarsening

    def extend_basis(self, q, snapshots, return_fourier = None):
        fourier_coefficients_list = []
        basis_before_enrichment = self.primal_reductor.bases["RB"]
        basis_length_before_enrichment = len(basis_before_enrichment)
        for U in snapshots:
            try:
                basis = self.primal_reductor.bases["RB"]
                product = self.primal_reductor.products.get('RB')
                copy_U = True             
                basis_length = len(basis)
                basis.append(U, remove_from_other=(not copy_U)) 
                basis, R = gram_schmidt(basis, product=product, return_R=True, atol=1e-13, rtol=1e-13, offset=basis_length, copy=False, check=False)
                # self.primal_reductor._check_orthonormality(basis, basis_length)
                self.primal_reductor.bases["RB"] = basis
                fourier_coefficients_list.append(R[:basis_length_before_enrichment,-1])
            except:
                 pass
        
        # update basis
        self.primal_reductor.bases["RB"] = basis
        self.RB = self.primal_reductor.bases["RB"]
        self.bases = dict(RB=self.RB)
        
        # extract fourier coefficients
        if return_fourier:
           return fourier_coefficients_list
       
    def set_new_RB(self, RB_new):
        self.RB = RB_new
        self.bases["RB"] = RB_new
        self.primal_reductor.bases["RB"] = RB_new
        return
    
    def coarse_basis(self,index):
        self.primal_reductor.bases["RB"] = self.primal_reductor.bases["RB"][index:]
        self.RB = self.primal_reductor.bases["RB"]
        self.bases = dict(RB=self.RB)
        return
        
    def coarse_basis_fourier(self,Fouriercoefficients_list):
        # coarsening technique T1 from [RB Pascoletti Serafini Multiobjective '22 Opt Paper]
        remove_tolerance = 1e-13
        listt = []
        for fourier in Fouriercoefficients_list:                               
            Teiler = sum(fourier**2)     
            res = []
            for i in range(len(fourier)):
                res.append( fourier[i]**2/Teiler)
            listt.append(res)                                                   
        listt = np.asarray(listt)                                               
        xis = np.max(listt,axis=0)                                             
        index = list((xis >= remove_tolerance).nonzero()[0])     
        index = index + [index[-1]+i+1 for i in range(len(Fouriercoefficients_list))]
        remove_index = list((xis < remove_tolerance).nonzero()[0])
        if remove_index == []:
           flag_remove = 0
           print('          NO COARSENING')
        else:
            flag_remove = 1
            print('          COARSE BASIS: indices to remove: {remove_index}')   
        numpy_basis = self.primal_reductor.bases["RB"].to_numpy()
        coarsed_numpy_basis = numpy_basis[index,:]
        new_basis = self.fom.solution_space.from_numpy(coarsed_numpy_basis)
        self.primal_reductor.bases["RB"] = new_basis
        self.RB = self.primal_reductor.bases["RB"]
        self.bases = dict(RB=self.RB)
        return flag_remove
    
#%% build rom/extend rom and project operators

    def get_primal_reductor(self):
        if self.reductor_type == "non_assembled":
            primal_reductor = NonAssembledCoerciveRBReductor(
                self.fom.primal_model,
                RB=self.RB,
                product=self.product,
                coercivity_estimator=self.coercivity_estimator)
        elif self.reductor_type == "simple_coercive":
            primal_reductor = SimpleCoerciveRBReductor(
                self.fom.primal_model,
                RB=self.RB,
                product=self.product,
                coercivity_estimator=self.coercivity_estimator)
        elif self.reductor_type == "hierarchical_non_assembled":
             pass
        elif self.reductor_type == "hierarchical_assembled":
             pass
        return primal_reductor

    def extend_rom(self, q, snapshots, snapshots_hierarchical = None, iteration = 0):
              
        ###### 1. update basis
        if self.RB is not None:
            len_old_RB = len(self.RB)
        else:
            len_old_RB = 0
        
        for snapshot in snapshots:
            try:
                self.primal_reductor.extend_basis(snapshot)
            except:
                pass   
        self.RB = self.primal_reductor.bases["RB"]
        self.bases = dict(RB=self.RB)

        ###### 2. project stuff
        projected_primal_operators, dual_rhs_projection = self.project_operators(len_old_RB)
        # project partial dual rhs
        opt_data_copy = deepcopy(self.opt_data)
        opt_data_copy["mass_u_noise"] = dual_rhs_projection

        ####### 3. assemble estimators
        self.estimators = self.assemble_error_estimator()
        # build primal model
        self.primal_rom = StationaryModel(error_estimator=self.estimators['primal'], **projected_primal_operators)
    
        # 4. extend rom_IP
        rom_IP = StationaryModelIP(
            self.primal_rom,
            self.fom.primal_data,
            opt_data_copy,
            projected_primal_operators['products'],
            self.estimators,
            hierarchical_model = None,
            name="rom_IP",
            boundary_info=self.fom.boundary_info)
        
        return rom_IP

    def project_operators(self, len_old_RB):
        
        # NOTE:
        # The code for this function is a modification of the pyMOR code
        # (see https://github.com/pymor/pymor/blob/2023.1.0/src/pymor/reductors/coercive.py).
        
        fom = self.fom
        RB = self.RB
        projected_primal_operators = {
            'operator':          project(fom.operator, RB, RB),
            'rhs':               project(fom.rhs, RB, None),
            'products':          {k: project(v, RB, RB) for k, v in fom.products.items()},
        }  
        # project dual/expand dual rhs
        dual_rhs_projection = project(
            VectorOperator(self.fom.opt_data["mass_u_noise"]), self.RB, None
        ).as_range_array()
        # update dual rhs for error estimating
        if len_old_RB == 0 and self.reductor_type == 'simple_coercive':
            self.dual_rhs = self.build_dual_rhs()
        elif self.reductor_type == 'simple_coercive':
            self.dual_rhs = self.update_dual_rhs(len_old_RB, len(self.RB))
        else:
            self.dual_rhs = None
        return projected_primal_operators, dual_rhs_projection
    
#%% build dual rhs for assembly of the error estimate

    def build_dual_rhs(self):
        RB = self.bases['RB']
        rhs_operators = [VectorOperator(self.opt_data["mass_u_noise"])]
        rhs_coefficients = [1.]
        bilinear_part = self.fom.l2_product
        for i in range(len(RB)):
            u = RB[i]
            rhs_operators.append(VectorOperator(bilinear_part.apply(u, mu=None)))
            rhs_coefficients.append(- ProjectionParameterFunctional('u_coefficients', len(RB), i))
        return LincombOperator(rhs_operators, rhs_coefficients)

    def update_dual_rhs(self, len_old_RB, len_new_RB):
        length_RB_extension = len_new_RB - len_old_RB
        if length_RB_extension == 0:
            return self.dual_rhs
        RB = self.bases['RB']
        rhs_operators = [self.dual_rhs.operators[0]]
        rhs_coefficients = [self.dual_rhs.coefficients[0]]
        for i, (ops, coefs) in enumerate(zip(self.dual_rhs.operators[1:], self.dual_rhs.coefficients[1:])):
            rhs_operators.append(ops)
            rhs_coefficients.append(- ProjectionParameterFunctional('u_coefficients', len(RB), i))
        bilinear_part = self.fom.l2_product
        for i in range(len_old_RB,len_new_RB):
            u = RB[i]
            rhs_operators.append(VectorOperator(bilinear_part.apply(u, mu=None)))
            rhs_coefficients.append(- ProjectionParameterFunctional('u_coefficients', len(RB), i))
        return LincombOperator(rhs_operators, rhs_coefficients)
    
#%% assemble error estimator/residuals

    def assemble_error_estimator(self):
        estimators = {}      
        if self.reductor_type == 'simple_coercive':
            estimators["primal"] = self.assemble_primal_error_estimator()
            estimators["dual_residual"] = self.assemble_dual_residual()         
        else:
            estimators["primal"] = self.primal_reductor.assemble_error_estimator()
            estimators["dual_residual"] = UnassembledDualResidualRBEstimator(None, self, None)
            #estimators["linearized_state"] = LinearizedStateRBEstimator(self.primal_rom, self)
            #estimators["linearized_dual_residual"] = LinearizedDualResidualRBEstimator(self.primal_rom, self)
        return estimators

    def assemble_primal_error_estimator(self):
        
        # NOTE: 
        # The code for this function is a modification of the pyMOR code
        # (see https://github.com/pymor/pymor/blob/2023.1.0/src/pymor/reductors/coercive.py).
        
        ####### 1. assemble primal error estimator
                fom, RB = self.fom, self.bases['RB']
                if self.old_reductor:
                    extends = self.old_reductor.primal_extends
                else:
                    extends = None
                
                if extends:
                    old_RB_size = extends[0]
                    old_data = extends[1]
                else:
                    old_RB_size = 0
                    
                # compute data for error estimator
                space = fom.operator.source
        ### compute all components of the residual matrix
                ## 1. rhs if it is parametric
                if extends:
                    R_R, RR_R = old_data['R_R'], old_data['RR_R']
                elif not fom.rhs.parametric:
                    R_R = space.empty(reserve=1)
                    RR_R = space.empty(reserve=1)
                    self.append_riesz_vector(fom.rhs.as_range_array(), R_R, RR_R)
                else:
                    R_R = space.empty(reserve=len(fom.rhs.operators))
                    RR_R = space.empty(reserve=len(fom.rhs.operators))
                    for op in fom.rhs.operators:
                        self.append_riesz_vector(op.as_range_array(), R_R, RR_R)
    
                ## 2. fom operator if it is parametric
                if len(RB) == 0:
                    R_Os = [space.empty()]
                    RR_Os = [space.empty()]
                elif not fom.operator.parametric:
                    R_Os = [space.empty(reserve=len(RB))]
                    RR_Os = [space.empty(reserve=len(RB))]
                    for i in range(len(RB)):
                        self.append_riesz_vector(-fom.operator.apply(RB[i]), R_Os[0], RR_Os[0])
                else: # hier sind wir....
                    R_Os = [space.empty(reserve=len(RB)) for _ in range(len(fom.operator.operators))]
                    RR_Os = [space.empty(reserve=len(RB)) for _ in range(len(fom.operator.operators))]
                    
                    # get older rietzrepresentatives if available
                    if old_RB_size > 0: 
                        for R_O, RR_O, old_R_O, old_RR_O in zip(R_Os, RR_Os,
                                                                    old_data['R_Os'], old_data['RR_Os']):
                            R_O.append(old_R_O)
                            RR_O.append(old_RR_O)
                    
                    # all rietzrepres for new affine components
                    if self.old_basis_Q_len == 0: 
                       for i in range(len(RB)):
                           self.append_riesz_vector(-fom.operator.operators[0].apply(RB[i]), R_Os[0], RR_Os[0])
                    else: 
                        for op, R_O, RR_O in zip(fom.operator.operators[:self.old_basis_Q_len+1], R_Os, RR_Os):  
                            for i in range(old_RB_size, len(RB)):
                                self.append_riesz_vector(-op.apply(RB[i]), R_O, RR_O)
                                
                    for j in range(self.old_basis_Q_len + 1, self.old_basis_Q_len + self.basis_Q_diff + 1): 
                        for i in range(len(RB)):
                            self.append_riesz_vector(-fom.operator.operators[j].apply(RB[i]), R_Os[j], RR_Os[j])
    
                # compute Gram matrix of the residuals
                R_RR = RR_R.inner(R_R)                                                   
                R_RO = np.hstack([RR_R.inner(R_O) for R_O in R_Os])                    
                R_OO = np.vstack([np.hstack([RR_O.inner(R_O) for R_O in R_Os]) for RR_O in RR_Os]) 
                estimator_matrix = np.empty((len(R_RR) + len(R_OO),) * 2)
                estimator_matrix[:len(R_RR), :len(R_RR)] = R_RR                       
                estimator_matrix[len(R_RR):, len(R_RR):] = R_OO                       
                estimator_matrix[:len(R_RR), len(R_RR):] = R_RO                        
                estimator_matrix[len(R_RR):, :len(R_RR)] = R_RO.T                     
                estimator_matrix = NumpyMatrixOperator(estimator_matrix)
    
                # output estimate
                primal_error_est = SimpleCoerciveRBEstimator(estimator_matrix, self.coercivity_estimator,
                                                            output_estimator_matrices = None, output_functional_coeffs = None)
                # safe extends in this reductor
                self.primal_extends = (len(RB), dict(R_R=R_R, RR_R=RR_R, R_Os=R_Os, RR_Os=RR_Os))
                
                return primal_error_est
            
    def assemble_dual_residual(self):
        
        # NOTE: 
        # The code for this function is a modification of the pyMOR code
        # (see https://github.com/pymor/pymor/blob/2023.1.0/src/pymor/reductors/coercive.py).
        
        RR_Os = self.primal_extends[-1]['RR_Os']
        R_Os = self.primal_extends[-1]['R_Os']
        if self.old_reductor:
            extends = self.old_reductor.dual_extends
            len_old_RB = extends[0]
        else:
            extends = None
        space = R_Os[0].space
        R_R = space.empty(reserve=len(self.dual_rhs.operators))
        RR_R = space.empty(reserve=len(self.dual_rhs.operators))
        if extends:
            R_R.append(extends[-1]['R_R'])
            RR_R.append(extends[-1]['RR_R'])
            for op in self.dual_rhs.operators[len_old_RB+1:]:
                self.append_riesz_vector(op.as_range_array(), R_R, RR_R)
        else:
            for op in self.dual_rhs.operators:
                self.append_riesz_vector(op.as_range_array(), R_R, RR_R)
            
        # compute Gram matrix of the residuals
        R_RR = RR_R.inner(R_R)
        R_RO = np.hstack([RR_R.inner(R_O) for R_O in R_Os])
        R_OO = np.vstack([np.hstack([RR_O.inner(R_O) for R_O in R_Os]) for RR_O in RR_Os])

        estimator_matrix = np.empty((len(R_RR) + len(R_OO),) * 2)
        estimator_matrix[:len(R_RR), :len(R_RR)] = R_RR
        estimator_matrix[len(R_RR):, len(R_RR):] = R_OO
        estimator_matrix[:len(R_RR), len(R_RR):] = R_RO
        estimator_matrix[len(R_RR):, :len(R_RR)] = R_RO.T
        
        estimator_matrix = NumpyMatrixOperator(estimator_matrix)
        error_estimator = SimpleCoerciveRBEstimator(estimator_matrix, None, None, None)
        dual_intermediate_fom = self.fom.primal_model.with_(rhs=self.dual_rhs)
        
        # save dual extends
        self.dual_extends = (len(self.RB), dict(R_R = R_R, RR_R = RR_R))
        
        return DualResidualRBEstimator(self, error_estimator, dual_intermediate_fom)
    
    def riesz_representative(self,U):
        if self.primal_reductor.products['RB'] is None:
            return U.copy()
        else:
            return self.primal_reductor.products['RB'].apply_inverse(U)
        
    def append_riesz_vector(self, U, R, RR):
        RR.append(self.riesz_representative(U), remove_from_other=True)
        R.append(U, remove_from_other=True)

#%% class for dual residual

class DualResidualRBEstimator(ImmutableObject):
    def __init__(self, reductor, error_estimator, dual_intermediate_fom):
        self.__auto_init(locals())
        self.primal_fom = reductor.fom.primal_model
        self.fom = reductor.fom
        
    def estimate_error(self, u_r, p_r, q):
        return self.estimate_error_simple_coercive(u_r, p_r, q)
 
    def estimate_error_simple_coercive(self, u_r, p_r, q):
        q_with_us = np.append(q, u_r.to_numpy()[0])
        q_with_us = self.dual_intermediate_fom.parameters.parse(q_with_us)
        dual_res_riesz_norm_ = self.error_estimator.estimate_error(p_r, q_with_us, self.dual_intermediate_fom)
        return dual_res_riesz_norm_[0] 

#%% non-assembled reductor and error estimator classes

class UnassembledDualResidualRBEstimator(ImmutableObject):
    def __init__(self, primal_rom, reductor, error_estimator):
        self.__auto_init(locals())
        self.primal_fom = reductor.fom.primal_model
        self.fom = reductor.fom
        
    def estimate_error(self, u_r, p_r, q):
        return self.estimate_error_non_assembled(u_r, p_r, q)
    
    def estimate_error_non_assembled(self, u_r, p_r, q):
        q = self.reductor.fom.parameters.parse(q)
        u = self.reductor.primal_reductor.reconstruct(u_r)
        p = self.reductor.primal_reductor.reconstruct(p_r)
        dual_rhs = (
            -self.primal_fom.l2_product.apply_adjoint(u)
            + self.fom.opt_data["mass_u_noise"])
        dual_residual = self.primal_fom.operator.apply(p, q) - dual_rhs
        riesz = self.reductor.product.apply_inverse(dual_residual)      # get riesz representative w.r.t. product
        dual_res_riesz_norm = np.sqrt(self.reductor.product.apply2(riesz, riesz))
        return dual_res_riesz_norm[0,0]

class LinearizedStateRBEstimator(ImmutableObject):
      def __init__(self, primal_rom, reductor):
          self.__auto_init(locals())
          self.primal_fom = reductor.fom.primal_model
          self.fom = reductor.fom

      def estimate_error(self, B_u_r, u_lin_r, d,q):
          d = self.primal_rom.parameters.parse(d)
          q = self.primal_rom.parameters.parse(d)
          u_lin = self.reductor.primal_reductor.reconstruct(u_lin_r)
          lin_rhs = self.reductor.primal_reductor.reconstruct(B_u_r.assemble(d).as_range_array())
          
          lin_residual = self.primal_fom.operator.apply(u_lin,q) - lin_rhs
          riesz = self.reductor.product.apply_inverse(
              lin_residual
          )  # get rietz representative w.r.t. product
          lin_res_riesz_norm = np.sqrt(
              self.reductor.product.apply2(riesz, riesz)
                                      )
          return lin_res_riesz_norm/self.reductor.coercivity_estimator(q) 
                     
class LinearizedDualResidualRBEstimator(ImmutableObject):
      def __init__(self, primal_rom, reductor):
          self.__auto_init(locals())
          self.primal_fom = reductor.fom.primal_model
          self.fom = reductor.fom
          
      def estimate_error(self, p_lin_r, u_lin_r, u_r, d,q):
          q = self.primal_rom.parameters.parse(q)
          d = self.primal_rom.parameters.parse(d)
          u_lin = self.reductor.primal_reductor.reconstruct(u_lin_r)
          p_lin = self.reductor.primal_reductor.reconstruct(p_lin_r)
          u = self.reductor.primal_reductor.reconstruct(u_r)
          dual_rhs = (
              -self.primal_fom.l2_product.apply_adjoint(u_lin+u)
              + self.fom.opt_data["mass_u_noise"]
          )
          dual_residual = self.primal_fom.operator.apply(p_lin, q) - dual_rhs
          riesz = self.reductor.product.apply_inverse(
              dual_residual
          )  # get rietz representative w.r.t. product
          dual_res_riesz_norm = np.sqrt(
              self.reductor.product.apply2(riesz, riesz)
          )
          return dual_res_riesz_norm  # no coercivity constant here

class NonAssembledCoerciveRBReductor(StationaryRBReductor):
    def __init__(self,
                fom,
                RB=None,
                product=None,
                coercivity_estimator=None,
                check_orthonormality=None,
                check_tol=None):
        assert fom.operator.linear and fom.rhs.linear
        assert isinstance(fom.operator, LincombOperator)
        assert all(not op.parametric for op in fom.operator.operators)
        if fom.rhs.parametric:
            assert isinstance(fom.rhs, LincombOperator)
            assert all(not op.parametric for op in fom.rhs.operators)
        super().__init__(
                        fom,
                        RB,
                        product=product,
                        check_orthonormality=check_orthonormality,
                        check_tol=check_tol)
        self.coercivity_estimator = coercivity_estimator

    def assemble_error_estimator(self):
        return NonAssembledCoerciveRBEstimator(self.fom, self.products["RB"], self)

    def assemble_estimator_for_subbasis(self, dims):
        return self._last_rom.error_estimator.restricted_to_subbasis(
            dims["RB"], m=self._last_rom)

class NonAssembledCoerciveRBEstimator(ImmutableObject):
    def __init__(self, fom, product, reductor):
        self.__auto_init(locals())

    def estimate_error(self, u, q, rom):
        u = self.reductor.reconstruct(u)                                        
        riesz = self.product.apply_inverse(                                     
            self.fom.operator.apply(u, q) - self.fom.rhs.as_vector(q))
        riesz_norm = np.sqrt(
            self.product.apply2(riesz, riesz))
        
        return riesz_norm/self.reductor.coercivity_estimator(q)