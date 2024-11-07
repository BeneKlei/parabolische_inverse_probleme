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
# Description: this file contains the implementation of the Qr-Vr TR IRGNM.

import numpy as np
from timeit import default_timer as timer
from reductor import CoerciveRBReductorIP
from pymor.core.base import BasicObject
from IRGNM import update_Q_space

#%% TR Surrogate

class IRGNM_TRSurrogate(BasicObject):
    def __init__(self, reductor, initial_guess, initial_snapshots = None, snapshots_hierarchical = None,
                 error_aware_TR = True, initial_steplength_AGC = None):
        self.__auto_init(locals())
        self.fom = reductor.fom
        if initial_snapshots is None:
            u, p, A_q = self.fom.solve_and_assemble(initial_guess)
            initial_snapshots = [u,p]
        if snapshots_hierarchical is None:
            self.hierarchical = False
        else: 
            self.hierarchical = True
        self.rom = reductor.extend_rom(initial_guess, initial_snapshots, snapshots_hierarchical, 0)
        self.new_reductor = None
        self.new_rom = None
        self.q_0 = self.fom.opt_data['q_0']                                 
        self.q_product = self.fom.Q_product
        self.hierarchical_constant = 0.001
        if initial_steplength_AGC is None:
            initial_steplength_AGC = 1
            
    # outputs, gradient
    def fom_output(self, q, u = None, Tikonov = 0):                                        
        if u is None:
            u, p, A_q = self.fom.solve_and_assemble(q)  
        output_res = self.fom.nonlin_output_res(u)
        if Tikonov == 0:
            return 0.5*output_res**2
        else:
            return 0.5*output_res**2 +  0.5*Tikonov*self.q_product(q-self.q_0,q-self.q_0)
        
    def rom_output(self, q, u_r=None, Tikonov = 0):                                         
        if u_r is None:
            u_r, p_r, A_q_r = self.rom.solve_and_assemble(q)                        
        output_res_r = self.rom.nonlin_output_res(u_r)  
        if Tikonov == 0:
            return 0.5*output_res_r**2
        else:
            return 0.5*output_res_r**2 +  0.5*Tikonov*self.q_product(q-self.q_0,q-self.q_0)
        
    def new_rom_output(self, q, u_r=None, Tikonov = 0):
        assert self.new_rom is not None, 'No new ROM found. Did you forget to call surrogate.extend()?'
        if u_r is None:
            u_r, p_r, A_q_r = self.new_rom.solve_and_assemble(q)  
        output_res_r = self.new_rom.nonlin_output_res(u_r)
        if Tikonov == 0:
            return 0.5*output_res_r**2
        else:
            return 0.5*output_res_r**2 +  0.5*Tikonov*self.q_product(q-self.q_0,q-self.q_0)
    
    def rb_size(self):
        return len(self.reductor.bases['RB'])
    
    # estimate  
    def estimate_output_error(self, q, u_r, p_r):
        return self.rom.estimate_output(u_r, p_r, q)

    def estimate_hierarchical_output(self, q, output_small = None):
        return self.rom.estimate_hierarchical_output(q, output_small, self.hierarchical_constant)
    
    def estimate_state(self, q, u_r = None):
        return self.rom.estimate_state(q,u_r)

    # extend, accept and reject
    def extend(self, q, snapshots = None, snapshots_hierarchical = None, iteration = 0):
        with self.logger.block('Trying to extend the basis...'):
            if snapshots is None:              
                u, p, A_q = self.fom.solve_and_assemble(q)
                snapshots = [u, p]
            if snapshots_hierarchical is None:
                self.hierarchical = False
            else: 
                self.hierarchical = True
            self.new_rom = self.reductor.extend_rom(q, snapshots, snapshots_hierarchical, iteration = iteration) 
            self.new_reductor = self.reductor # Deepcopy?

    def accept(self):
        assert self.new_rom is not None, 'No new ROM found. Did you forget to call surrogate.extend()?'
        self.rom = self.new_rom
        self.reductor = self.new_reductor
        self.new_rom = None
        self.new_reductor = None

    def reject(self):
        self.new_rom = None
        self.new_reductor = None
        
    # get acg point
    def get_agc(self,q, gradient, radius, beta, old_rom_output, Tikonov = 0, relax_factor = 0, iteration = 1, q_r = None, gradient_r = None):
        
        d = -gradient_r       
        u, p, A_q, q_AGC, est_output_AGC, counter_armijo, steplength, output_res_AGC, TR_condition_break = armijo_type_linesearch(
        q_r, d, self.rom, radius, self, beta, old_rom_output, relax_factor = relax_factor, iteration = iteration, compute_AGC = True, gradient = gradient_r)
        
        return q_AGC, est_output_AGC, output_res_AGC, TR_condition_break, u, p, counter_armijo

#%% Qr-Vr TR IRGNM

def Qr_Vr_TR_IRGNM(parameter_space, fom_IP, TR_IRGNM_setup, q0, tol = 1e-6, initial_Qsnapshots = [] ):

    #===== read setup and input checks ========================================
    # read IRGNM setup
    error_aware_TR = True
    hierarchical = False
    solver_option = 'direct'
    snapshot_invert = True
    tau = TR_IRGNM_setup["tau"]
    Inner_IRGNM_setup = TR_IRGNM_setup['Inner_IRGNM_setup']
    shrink_factor = TR_IRGNM_setup['shrink_factor']
    beta = TR_IRGNM_setup['beta']
    radius = TR_IRGNM_setup['radius']
    miniter = TR_IRGNM_setup['miniter']
    maxiter = TR_IRGNM_setup['maxiter']
    radius_tol = TR_IRGNM_setup['radius_tol']
    noise_level = TR_IRGNM_setup["noise_level"]
    reductor_type =  TR_IRGNM_setup["reductor_type"]
    display = TR_IRGNM_setup['display_iteration_process']
    outer_regularization = TR_IRGNM_setup['outer_regularization']
    alpha0 = TR_IRGNM_setup['alpha0']  
    relax_factor = TR_IRGNM_setup['relax_factor'] 
    assert shrink_factor != 0.

    #===== initialize ============================================================
    print('=============================================================================================================')
    print(f'Starting Qr-Vr TR IRGNM with total tolerance {0.5*(tol**2 + (tau * noise_level)**2)}')
    start_time = timer()
    q = q0.copy()
    fom_IP.check_parameter_bounds(q)
    u, p, A_q = fom_IP.solve_and_assemble(q, solver_option = solver_option)                                  
    start_time_B = timer()
    B_u = fom_IP.assemble_B_u(u)
    time_B_u = timer() - start_time_B
    gradient = fom_IP.gradient(B_u, p, q, tikonov_parameter = None)             
    q_snapshots = [q]
    if not fom_IP.Q_norm(q-fom_IP.opt_data['q_0'])< 1e-13:
        q_snapshots.append(fom_IP.opt_data['q_0'])
    number_grad_invert_fom_solves = 0
    if snapshot_invert and 1:
       gradient_obd = fom_IP.Q_prod_py.apply_inverse(fom_IP.Q_vector_space.from_numpy(gradient)).to_numpy()[0]
       q_snapshots.append(gradient_obd)
       gradient_norm = np.sqrt(gradient_obd.T@gradient)
       number_grad_invert_fom_solves += 1
    else: 
       q_snapshots.append(gradient)
       gradient_norm = fom_IP.Q_norm(gradient)

    # initialize Q-FOM
    basis_Q = fom_IP.Q_vector_space.empty()
    Qfom_IP, basis_Q, basis_diff_Q, old_basis_len_Q, coarse_flag, _ = update_Q_space(fom_IP, None,
                                                                               initial_Qsnapshots + q_snapshots,
                                                                               basis_Q, 0)
    # project q in basis
    q_r = basis_Q.to_numpy().dot(fom_IP.prod_matrix.dot(q))

    ## create reductor and surrogate
    old_data = {'old_rom': None, 'old_reductor': None, 'old_basis_Q_len': old_basis_len_Q, 'basis_diff_Q': basis_diff_Q}
    Qreductor = CoerciveRBReductorIP(Qfom_IP,
                            RB = None,
                            coercivity_estimator=Qfom_IP.opt_data['coercivity_q'],
                            product = fom_IP.opt_data["V_product"],
                            reductor_type = reductor_type,
                            name = 'QCoerciveRBReductorIP',
                            old_data = old_data)
    if hierarchical:
       d = -gradient
       u_lin = fom_IP.lin_state(q, d, u = u, A_q = A_q, B_u = B_u)
       p_lin = fom_IP.lin_adjoint(q , u = u, p = p, u_lin = u_lin,
                       A_q = A_q , B_u = B_u, include_rhs = True)
       snapshots_hierarchical = [u_lin, p_lin]
    else:
       snapshots_hierarchical = None
    initial_steplength_AGC = 0.5*fom_IP.Q_norm(gradient)**(-1)
    surrogate = IRGNM_TRSurrogate(Qreductor, q, initial_snapshots=[u, p],
                                  snapshots_hierarchical = snapshots_hierarchical, 
                                  error_aware_TR = error_aware_TR, 
                                  initial_steplength_AGC = initial_steplength_AGC)
    ## initialize iteration
    if outer_regularization:
        outer_alpha = alpha0                                                       
    else:
        outer_alpha = 0
    u_r, p_r, A_q_r = surrogate.rom.solve_and_assemble(q_r)
    old_rom_output = surrogate.rom_output(q_r, u_r, Tikonov = outer_alpha)
    old_fom_output =  surrogate.fom_output(q, u, Tikonov = outer_alpha) 
    len_old_RB = surrogate.rb_size()
    q_norm = fom_IP.Q_norm(q)
    get_new_AGC_point = True
    
    ## initialize iteration history
    history = {}                                                                  
    history['subproblem_history'] = []
    update_norms = []
    output_res = [old_fom_output]
    history['time1e-5'] = 0
    history['time_steps'] = [0]
    history['time_assemb_B_u'] = [time_B_u] 
    accepted_output_res = []
    history['armijo_iter'] = []
    history['number_fom_B_operator_applications'] = 1
    # history['qs'] = [q.copy()]                                                     
    # history['q_rs'] = [q_r.copy()]                                                 
    history['initial_alphas'] = []
    history['number_fom_solves'] = 2 + number_grad_invert_fom_solves
    if hierarchical:
        history["fom_solves_for_building_hierarchical_error_est"] = 2
    history['fom_solves_for_assembling_residual_error_est'] = [2 + (len(basis_Q) +1)*float(surrogate.rb_size())]
    history['fom_solves_for_unassembled_residual_error_est'] = []  
    history['true_fom_solves_for_residual_error_est'] = [2 + (len(basis_Q) +1)*float(surrogate.rb_size())]  
    history["accepted_rejected_string"] = []
    history["non_regularized_gradient_norm"] = [gradient_norm]
    switched_at_iteration = []
    
#============= Outer iteration ================================================
    iteration = 0                                                              
    print(f'k = {iteration:<3}, radius = {radius:1.2e}, output_res = {np.sqrt(2*old_fom_output):3.4e}, norm_grad = {history["non_regularized_gradient_norm"][-1]:3.4e}, len Q_basis = {len(basis_Q)}, len RB = {len_old_RB}, FOM solves: {history["number_fom_solves"]}')
    
    while True:
            
            # test if error est needs to be assembled or cimputed online
            if iteration > 3 and  history["fom_solves_for_unassembled_residual_error_est"][-1] < history["fom_solves_for_assembling_residual_error_est"][-1]:
               switched_at_iteration.append(iteration)
               reductor_type = 'non_assembled'
               print('Switch to online error est')
        
            #======== Convergence checks ======================================
            if not np.isfinite(q_norm):                                        
                print('DIVERGENCE')
                return
            if iteration >= miniter:
                if old_fom_output < 0.5*(noise_level*tau+tol)**2:
                    history['flag'] = f'Qr-Vr TR IRGNM converged after iteration {iteration} with output res = {np.sqrt(2*old_fom_output)}, time {history["time"]}'
                    break
                if iteration >= maxiter:
                    history['time'] = timer() - start_time
                    history['flag'] = f'Qr-Vr TR IRGNM reached maxiter of {iteration} with output res = {np.sqrt(2*old_fom_output)}, time {history["time"]}'
                    break
            rejected = False
            iteration += 1                                                     
                
            #========= Compute AGC Point/Landweber step =======================
            if 1: 
                print('         Get AGC point:')
                if get_new_AGC_point:
                   B_u_r = surrogate.rom.assemble_B_u(u_r)
                   gradient_r = surrogate.rom.gradient(B_u_r, p_r, q_r, tikonov_parameter = None) 
          
                   q_AGC, est_output_AGC, output_res_AGC, TR_condition_break, u_AGC, p_AGC, counter_armijo = surrogate.get_agc(
                       q, gradient, radius, beta, old_rom_output, Tikonov = outer_alpha, relax_factor = relax_factor, iteration = iteration, 
                       q_r = q_r,
                       gradient_r = gradient_r )
                   compare_output = output_res_AGC
                   q_init = q_AGC
            else:
                   TR_condition_break = False
                   compare_output = old_rom_output
                   q_init = q_r
                   counter_armijo = 0
            
            #========= Solve the subproblem using IRGNM =======================
            old_q = q.copy()
            old_q_r = q_r.copy()
            inner_tol = tol
            Inner_IRGNM_setup["noise_level"] = noise_level                      
            # Inner_IRGNM_setup["tau"] = tau                                       
            inner_init_guess = q_init.copy()                                    
            history['initial_alphas'].append(alpha0)
            if TR_condition_break:                                             
                q = q_AGC
                u_r_current, p_r_current, estimate_output, current_output = u_AGC, p_AGC, est_output_AGC, output_res_AGC
                sub_history = {'flag': 'used AGC', 'Alpha': [alpha0], 'counter_error_est': 0}
            else:                                                                
                q_r, sub_history = INNER_IRGNM_linear_tikonov(Inner_IRGNM_setup, surrogate.rom, inner_tol, inner_init_guess, radius, surrogate, beta, alpha0, relax_factor = relax_factor, iteration = iteration)
                u_r_current, p_r_current, A_q_r_current = surrogate.rom.solve_and_assemble(q_r)
                current_output = surrogate.rom_output(q, u_r_current, Tikonov = outer_alpha)
                estimate_output = sub_history['estimates'][-1]
               
            history['fom_solves_for_unassembled_residual_error_est'].append(sub_history['counter_error_est'] + counter_armijo)
            # Reconstruct q
            q = basis_Q.lincomb(q_r).to_numpy()[0]   
            fom_IP.check_parameter_bounds(q)
            
            #======== acceptance checks and radius modification ===============
            if current_output + estimate_output < compare_output + relax_factor/iteration and error_aware_TR:  
            ### sufficient criteria for acceptance
                string = '1. accepted'
                u_current, p_current, A_q_current = fom_IP.solve_and_assemble(q, solver_option = solver_option)         # get new snapshots
                current_fom_output = surrogate.fom_output(q_r, u_current, Tikonov = outer_alpha)
                history['number_fom_solves'] += 2
                
                # check for convergence
                if current_fom_output < 0.5*(noise_level*tau+tol)**2:
                    history['time'] = timer() - start_time
                
                # extend Q space, Q-FOM
                start_time_B = timer()
                B_u_current = fom_IP.assemble_B_u(u_current)
                history['time_assemb_B_u'].append( timer() - start_time_B ) 
                gradient = fom_IP.gradient(B_u_current, p_current, q, tikonov_parameter = None)
                history['number_fom_B_operator_applications'] += 1
                q_snapshots = []
                if snapshot_invert and 1:
                   gradient_obd = fom_IP.Q_prod_py.apply_inverse(fom_IP.Q_vector_space.from_numpy(gradient)).to_numpy()[0]
                   q_snapshots.append(gradient_obd)
                   gradient_norm = np.sqrt(gradient_obd.T@gradient)
                   history['number_fom_solves'] += 1
                else: 
                   q_snapshots.append(gradient)
                   gradient_norm = fom_IP.Q_norm(gradient)
                Qfom_IP, basis_Q, basis_diff_Q, old_basis_len_Q, coarse_flag, _ = update_Q_space(fom_IP, Qfom_IP, q_snapshots, basis_Q, iteration,  other_snapshots = [q, q0.copy()])
                
                # update reductor 
                old_data = {'old_rom': surrogate.rom, 'old_reductor': surrogate.reductor, 'old_basis_Q_len': old_basis_len_Q, 'basis_diff_Q': basis_diff_Q}
                Qreductor = CoerciveRBReductorIP(Qfom_IP,
                                        RB = surrogate.reductor.RB,
                                        coercivity_estimator = Qfom_IP.opt_data['coercivity_q'],
                                        product = fom_IP.opt_data["V_product"],
                                        reductor_type = reductor_type,         
                                        name = 'QCoerciveRBReductorIP',
                                        old_data = old_data)
                surrogate.reductor = Qreductor
                surrogate.fom = Qfom_IP
                
                # project back into the basis
                q_r = basis_Q.to_numpy().dot(fom_IP.prod_matrix.dot(q))
                
                if hierarchical:
                    d = -gradient
                    u_lin = surrogate.fom.lin_state(q, d, u = u_current, A_q = A_q_current, B_u = B_u_current)
                    p_lin = surrogate.fom.lin_adjoint(q , u = u_current, p = p_current, u_lin = u_lin,
                                    A_q = A_q_current , B_u = B_u_current, include_rhs = True)
                    snapshots_hierarchical = [u_lin, p_lin]
                    history['fom_solves_for_building_hierarchical_error_est'] += 2
                else:
                    snapshots_hierarchical = None
                    
                # extend Q ROM
                surrogate.extend(q_r, [u_current, p_current], snapshots_hierarchical, iteration = iteration)
                
                # compute Q fom output
                fom_output_diff = old_fom_output - current_fom_output           # compute rho
                rom_output_diff = old_rom_output - current_output
                if fom_output_diff >= radius_tol * rom_output_diff:
                    radius /= shrink_factor               
                    string = '1. accepted and enlarged'
            elif current_output - estimate_output > compare_output + relax_factor/iteration and error_aware_TR:
            ###  sufficient criteria for rejection
                string = '2. rejected'
                rejected = True                                                 # reject new q
                radius *= shrink_factor                                         # shrink the radius
            else:    
            #### decide by using fom    
                u_current, A_q_current = fom_IP.solve_and_assemble_state(q, solver_option = solver_option)     # get new snapshots 
                history['number_fom_solves'] += 1               
                current_fom_output = 0.5*fom_IP.nonlin_output_res(u_current)**2
                current_output = current_fom_output
                
                # check for convergence
                if current_output < 0.5*(noise_level*tau+tol)**2:
                    history['time'] = timer() - start_time
                
                if current_output <= compare_output:
                    
                    p_current = fom_IP.adjoint( q, u_current, A_q_current)
                    history['number_fom_solves'] += 1
                    start_time_B = timer()
                    B_u_current = fom_IP.assemble_B_u(u_current)
                    history['time_assemb_B_u'].append( timer() - start_time_B ) 
                    gradient = fom_IP.gradient(B_u_current, p_current, q, tikonov_parameter = None)   
                    history['number_fom_B_operator_applications'] += 1
                    q_snapshots = []
                    if snapshot_invert and 1:
                       gradient_obd = fom_IP.Q_prod_py.apply_inverse(fom_IP.Q_vector_space.from_numpy(gradient)).to_numpy()[0]
                       q_snapshots.append(gradient_obd)
                       gradient_norm = np.sqrt(gradient_obd.T@gradient)
                       history['number_fom_solves'] += 1
                    else: 
                       q_snapshots.append(gradient)
                       gradient_norm = fom_IP.Q_norm(gradient)
                    
                    if hierarchical:
                        d = -gradient
                        u_lin = surrogate.fom.lin_state(q, d, u = u_current, A_q = A_q_current, B_u = B_u_current)
                        p_lin = surrogate.fom.lin_adjoint(q , u = u_current, p = p_current, u_lin = u_lin,
                                        A_q = A_q_current , B_u = B_u_current, include_rhs = True)
                        history['fom_solves_for_building_hierarchical_error_est'] += 2
                        snapshots_hierarchical = [u_lin, p_lin]
                    else:
                        snapshots_hierarchical = None
                        
                    # extend Q space, Q-FOM
                    Qfom_IP, basis_Q, basis_diff_Q, old_basis_len_Q, coarse_flag, _ = update_Q_space(fom_IP, Qfom_IP, q_snapshots, basis_Q, iteration, other_snapshots = [q, q0.copy()])
                    
                    # update reductor 
                    old_data = {'old_rom': surrogate.rom, 'old_reductor': surrogate.reductor, 'old_basis_Q_len': old_basis_len_Q, 'basis_diff_Q': basis_diff_Q}
                    Qreductor = CoerciveRBReductorIP(Qfom_IP,
                                            RB = surrogate.reductor.RB,
                                            coercivity_estimator = Qfom_IP.opt_data['coercivity_q'],
                                            product = fom_IP.opt_data["V_product"],
                                            reductor_type = reductor_type,      
                                            name = 'QCoerciveRBReductorIP',
                                            old_data = old_data)
                    surrogate.reductor = Qreductor
                    surrogate.fom = Qfom_IP
                    
                    # project back into the basis
                    q_r = basis_Q.to_numpy().dot(fom_IP.prod_matrix.dot(q)) 
                    
                    # extend QROM
                    surrogate.extend(q_r, [u_current,p_current], snapshots_hierarchical, iteration = iteration)                           
                    fom_output_diff = old_fom_output - current_fom_output
                    rom_output_diff = old_rom_output - current_output
                    string = '3. accepted'
                    if fom_output_diff >= radius_tol * rom_output_diff:     
                        radius /= shrink_factor                                 # increase the radius if the model confidence is high enough
                        string = '3. accepted and enlarged'
                else:
                    rejected = True
                    radius *= shrink_factor
                    string = '3. rejected'
                
            #========= handle parameter rejection =============================
            if not rejected:
                # acceptance
                old_rom_output = current_fom_output
                old_fom_output = current_fom_output
                q_norm = fom_IP.Q_norm(q)
                update_norms.append(fom_IP.Q_norm(q - old_q))
                
                # history['qs'].append(q.copy())
                # history['q_rs'].append(q_r.copy())
                history['subproblem_history'].append(sub_history)
                accepted_output_res.append(current_fom_output)
                history["non_regularized_gradient_norm"].append(gradient_norm)
                surrogate.accept()
                
                u_r, p_r, A_q_r = surrogate.rom.solve_and_assemble(q_r)
                u, p, A_q = u_current, p_current, A_q_current
                B_u =  B_u_current
                get_new_AGC_point = True
                sub_history_accepted = sub_history
                
                RB_len_diff = surrogate.rb_size() - len_old_RB
                len_old_RB = surrogate.rb_size()
                
                if not coarse_flag: # no coarsening
                    history['fom_solves_for_assembling_residual_error_est'].append(old_basis_len_Q*RB_len_diff + basis_diff_Q*surrogate.rb_size() + RB_len_diff)
                else: # coarsening, so build everything from scratch
                    history['fom_solves_for_assembling_residual_error_est'].append(basis_diff_Q*surrogate.rb_size())
                                
                # get next initial alpha
                if len(sub_history_accepted['Alpha']) >= 2:
                    alpha0 = sub_history_accepted['Alpha'][1]                  
                else: 
                    alpha0 = sub_history_accepted['Alpha'][0]
                alpha0 = max(alpha0, 1e-14)
                if outer_regularization:
                   outer_alpha = alpha0
            else:
                # rejection
                q = old_q
                q_r = old_q_r
                surrogate.reject()
                get_new_AGC_point = False
                if TR_condition_break:
                    get_new_AGC_point = True
                    
                # extend if twice rejected
                if len(history["accepted_rejected_string"])>2:                    
                    if 'rejected' in history["accepted_rejected_string"][-1] and 'rejected' in history["accepted_rejected_string"][-2] and 'rejected' in history["accepted_rejected_string"][-3]:
                        u_current, p_current, A_q_current = fom_IP.solve_and_assemble(q)
                        
                        # extend Q space, Q-FOM
                        start_time_B = timer()
                        B_u_current = fom_IP.assemble_B_u(u_current)
                        history['time_assemb_B_u'].append( timer() - start_time_B ) 
                        
                        gradient = fom_IP.gradient(B_u_current, p_current, q, tikonov_parameter = None)
                        history['number_fom_B_operator_applications'] += 1
                        q_snapshots = []
                        if snapshot_invert and 1:
                           gradient_obd = fom_IP.Q_prod_py.apply_inverse(fom_IP.Q_vector_space.from_numpy(gradient)).to_numpy()[0]
                           q_snapshots.append(gradient_obd)
                           gradient_norm = np.sqrt(gradient_obd.T@gradient)
                           history['number_fom_solves'] += 1
                        else: 
                           q_snapshots.append(gradient)
                           gradient_norm = fom_IP.Q_norm(gradient)
                                
                        Qfom_IP, basis_Q, basis_diff_Q, old_basis_len_Q, coarse_flag, _ = update_Q_space(fom_IP, Qfom_IP, q_snapshots, basis_Q, iteration,  other_snapshots = [q, q0.copy()])
                        
                        # update reductor 
                        old_data = {'old_rom': surrogate.rom, 'old_reductor': surrogate.reductor, 'old_basis_Q_len': old_basis_len_Q, 'basis_diff_Q': basis_diff_Q}
                        Qreductor = CoerciveRBReductorIP(Qfom_IP,
                                                RB = surrogate.reductor.RB,
                                                coercivity_estimator = Qfom_IP.opt_data['coercivity_q'],
                                                product = fom_IP.opt_data["V_product"],
                                                reductor_type = reductor_type,
                                                name = 'QCoerciveRBReductorIP',
                                                old_data = old_data)
                        surrogate.reductor = Qreductor
                        surrogate.fom = Qfom_IP
                        
                        # project back into the basis
                        q_r = basis_Q.to_numpy().dot(fom_IP.prod_matrix.dot(q))
                        surrogate.extend(q_r, [u_current, p_current], snapshots_hierarchical, iteration = iteration)  
                        surrogate.accept()
                        current_fom_output = 0.5*fom_IP.nonlin_output_res(u_current)**2
                        old_rom_output = current_fom_output
                        old_fom_output = current_fom_output
                        get_new_AGC_point = True
                        print('enriched after a few time of rejection.....')
                   
            # update history and display
            output_res.append(old_fom_output)
            history["accepted_rejected_string"].append(string)
            history['time_steps'].append(timer() - start_time)
            if display == 1:
               print(f'k = {iteration:<3}' + string + \
                     f' radius = {radius:1.2e}, '
                     + sub_history['flag'])
               print(f'     output_res = {np.sqrt(2*old_fom_output):3.4e}, norm_grad = {history["non_regularized_gradient_norm"][-1]:3.4e}, len Q_basis = {len(basis_Q)}, len RB = {len_old_RB}, FOM solves: {history["number_fom_solves"]}, FOM solves unassembled error est: {history["fom_solves_for_unassembled_residual_error_est"][-1]}, FOM solves assembling error est: {history["fom_solves_for_assembling_residual_error_est"][-1]}')

#============= Finalize Output ================================================
    print('--------------------- RESULTS ------------------------------------')
    #logger.info('')
    history['update_norms'] = np.array(update_norms)
    history['accepted_output_res'] = np.sqrt(2*np.array(accepted_output_res))
    history['output_res'] = np.sqrt(2*np.array(output_res))  
    history['k'] = iteration  
    history['time_steps'][-1] = history["time"]
    history['len_Q_basis'] = len(basis_Q)
    history['len_V_basis'] = len_old_RB
    del history['subproblem_history']
    history['true_fom_solves_for_residual_error_est'] = sum(history["fom_solves_for_unassembled_residual_error_est"][switched_at_iteration[0]:]) + sum(history["fom_solves_for_assembling_residual_error_est"][:switched_at_iteration[0]])
    history['q'] = q
    # print(f'Time to assemble B_u: {sum(history["time_assemb_B_u"])}')
    # print(f'Elapsed time to reach 1e-5 is {history["time1e-5"]:4.5f} seconds.')
    print(history['flag'])
    print(f'Elapsed time is {history["time"]:4.5f} seconds.')
    print(f'V reduced basis size is {surrogate.rb_size()} and Q basis size is {len(basis_Q)}')
    #print(history['initial_alphas'])
    print(f'FOM solves: {history["number_fom_solves"]}, true FOM solves error est: {history["true_fom_solves_for_residual_error_est"]}, FOM solves unassembled error est:  {sum(history["fom_solves_for_unassembled_residual_error_est"])}, FOM solves assembling error est:  {sum(history["fom_solves_for_assembling_residual_error_est"])}')
    # print(f'FOM solves: {history["number_fom_solves"]}, true FOM solves error est: {history["true_fom_solves_for_residual_error_est"]}, FOM solves unassembled error est:  {sum(history["fom_solves_for_unassembled_residual_error_est"])}, FOM solves assembling error est:  {sum(history["fom_solves_for_assembling_residual_error_est"])}')
    print('------------------------------------------------------------------')
    return q, history

#%%  inner IRGNM

from IRGNM import solve_lin_quad_ocp_with_cg

def armijo_type_linesearch(q, d, model, radius, surrogate, beta, output_current, Tikonov = 0, relax_factor = 0, iteration = 1, compute_AGC = False, gradient = None ):
    
    # initialize armijo linesearch
    TR_condition_break = False
    armijo_kappa = 1e-12
    max_count = 100
    
    # choose initial steplength:
    if compute_AGC: 
        steplength = surrogate.initial_steplength_AGC
    else:
        steplength = 1
    counter = 1
    skip_TR_conditions_from_now_on = False
    counter_fomsolve_unassembled_error_est = 0
    
    while True:  
          q_trial = q + steplength* d                                          
          u, p, A_q = model.solve_and_assemble(q_trial)
          output_trial = surrogate.rom_output(q_trial,u, Tikonov = Tikonov)
          # check armijo condition
          if gradient is None:
              Armijo_condition = ((output_trial - output_current) <= -armijo_kappa/steplength*(surrogate.fom.Q_norm(q_trial-q))**2)
          else:  # unconstrained armijo condition
              Armijo_condition = ((output_trial - output_current) <= armijo_kappa * steplength*surrogate.fom.Q_product(d, gradient))
         
          # check TR condition
          if not skip_TR_conditions_from_now_on:
              if surrogate.error_aware_TR:                                          
                  if surrogate.hierarchical:
                     output_est = surrogate.estimate_hierarchical_output(q_trial, output_trial)
                     # print(f'hierarchical est {output_est}')
                  else:
                     output_est = surrogate.estimate_output_error(q_trial, u, p)
                     # print(f'output est {output_est}')
                  TR_crit = output_est/output_trial 
                  counter_fomsolve_unassembled_error_est += 2
                  TR_condition = TR_crit <= radius + relax_factor/iteration
                  # print(f'TR_crit {TR_crit}')
              else:                                                                 
                  TR_crit = surrogate.fom.Q_norm(steplength*d)
                  TR_condition = (TR_crit <= radius + relax_factor/iteration)
                  output_est = 0 
                  
          if TR_condition and 1:
              skip_TR_conditions_from_now_on = True
            
          # check both conditions and terminate if it is satisfied
          condition = TR_condition and Armijo_condition
          if condition == True or counter >= max_count:
             break
          else:
              pass
              #print([TR_condition, Armijo_condition])
         
          # update if necessary
          steplength *= 0.5
          counter += 1
    
    # check if boundary stopping criteria is met
    if TR_crit >= beta*radius + relax_factor/iteration:
       TR_condition_break = True
       
    if counter == max_count:
       print(f'         Armijo did not find a good steplength after {counter:<3} tries. TR_cond = {TR_condition}, Armijo_cond = {Armijo_condition}')
    else:
       print(f'         Armijo find a good steplength after {counter:<3} tries. TR_cond = {TR_condition}, Armijo_cond = {Armijo_condition}, decrease: {(output_trial - output_current):3.4e}')
     
    return u, p, A_q, q_trial, output_est, counter_fomsolve_unassembled_error_est, steplength, output_trial, TR_condition_break

def INNER_IRGNM_linear_tikonov(IRGNM_setup,model_IP, tol, q0, radius, surrogate, beta, alpha0, relax_factor = 0, iteration = 1):
    
    # read IRGNM_setup and opt_data
    display = False
    tau = IRGNM_setup["tau"]
    k_max = IRGNM_setup["k_max"]
    noise_level = IRGNM_setup["noise_level"]
    Theta = IRGNM_setup['Theta']
    theta = IRGNM_setup['theta']

    # initialize
    start_time = timer()
    q = q0.copy()
    u, p, A_q = model_IP.solve_and_assemble(q)                                  
    B_u = model_IP.assemble_B_u(u)                                             
    output_res = surrogate.rom_output(q,u)                                     
    if surrogate.hierarchical:
       output_est = surrogate.estimate_hierarchical_output(q, output_res)
    else:
       output_est = surrogate.estimate_output_error(q, u, p)
    k = 0                                                                       
    alpha = alpha0
    TR_condition_break = (output_est/output_res) >= beta*radius
    history = {'output_res': [output_res], 'Alpha': [alpha], 'number_rom_solves': 2, 'steplength': [], 'counter_error_est': 0, 'estimates': [output_est]}
    print(f'        Starting Inner IRGNM with total tolerance {0.5*(tol**2 + (tau * noise_level)**2)}')
    # =========== LOOP ================================
    while output_res >= 0.5*(tol**2 + (tau * noise_level)**2) and (not TR_condition_break) and k < k_max:

        #============= Solve subproblem and get regularization parameter ======
        regularization_qualification = False
        count = 1
        Alpha = [alpha]
        lin_res_old = 1
        while count < 50:
            # =========== CG ================================
            d, hist_inner = solve_lin_quad_ocp_with_cg(IRGNM_setup, model_IP, u, p, q, A_q, B_u, alpha)
            u_q_d = model_IP.lin_state(q, d, u=u, A_q=A_q, B_u=B_u)
            lin_res = 0.5*model_IP.lin_output_res(u, u_q_d, None, None)**2
            condition_low = theta*output_res < lin_res
            condition_up = lin_res < Theta * output_res
            regularization_qualification = condition_low and condition_up #lin_res < Theta * output_res
            alpha = max(alpha / 2, 1e-14)
            Alpha.append(alpha)
            if regularization_qualification == True:
                regularization_choice_flag ='              Regularization search succesfull'
                break
            if count > 2:
                if lin_res_old -lin_res < 1e-14 or Alpha[-1] == Alpha[-2] == 1e-14 :
                    regularization_choice_flag = '              Reg search stopped'
                    break
            lin_res_old = lin_res
            count += 1
            
        if display == 1:
            print(regularization_choice_flag)
            print(f'        k = {k:<3}, lin_res {lin_res:3.3e} < output_res = {output_res:3.3e},' \
                  f' alpha = {alpha:1.2e} after {count:<2} tries, '
                  + hist_inner['exit_flag'])

        # =========== UPDATE ================================
        gradient = -hist_inner['rhs_cg']
        u, p, A_q, q_trial, est, counter_armijo, steplength, output_res, TR_condition_break = armijo_type_linesearch(
            q, d, model_IP, radius, surrogate, beta, output_res, relax_factor = relax_factor, iteration = iteration, gradient = gradient)
        if display == 1:
            print(f'              steplength {steplength:3.2e} accepted after {counter_armijo:<2} tries.')
        q = q_trial
        B_u = model_IP.assemble_B_u(u)
        
        # update history
        history['output_res'].append(output_res)
        history['Alpha'].append(alpha)
        history['steplength'].append(steplength)
        history['counter_error_est'] += counter_armijo
        history['estimates'].append(est)
        
        # stagnation check
        if k > 3:
           if abs(history['output_res'][-1] -  history['output_res'][-2]) <1e-12:
                stagnation_flag = True
                break
        k += 1
    
    # save and print history
    history['time'] = timer() - start_time
    history['k'] = k
    if k == k_max and TR_condition_break == False:
        history['flag'] = f'        Inner IRGNM reached maxit k = {str(k)}' \
                          f' with output_res = {np.sqrt(2*output_res):3.4e}'
    elif k < k_max and TR_condition_break == False:
        history['flag'] = f'        Inner IRGNM converged at k = {str(k)}' \
                          f' with output_res = {np.sqrt(2*output_res):3.4e}'
    elif TR_condition_break == True:
        history['flag'] = f'        Inner IRGNM TR boundary criterium triggered at k = {str(k)}' \
                          f' with output_res = {np.sqrt(2*output_res):3.4e}'
    elif stagnation_flag == True:
        history['flag'] = f'        Inner IRGNM TR stagnated at k = {str(k)}' \
                          f' with output_res = {np.sqrt(2*output_res):3.4e}'

    print(history['flag'])
    #print('------------------------------------------------------------------')
    #print(f'Elapsed time is {history["time"]:4.5f} seconds.')
    #print(f'Total FOM solves {history["number_rom_solves"]}.')
    #print(f'Regularization parameters {history["Alpha"]}.')
    return q, history