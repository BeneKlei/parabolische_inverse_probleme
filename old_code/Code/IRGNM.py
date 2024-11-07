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
# Description: this file contains the implementation of the FOM IRGNM and the Qr IRGNM. 

import numpy as np
from helpers import conjugate_gradient, plot_and_save_iterate
from copy import deepcopy
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.algorithms.projection import project
from pymor.algorithms.gram_schmidt import gram_schmidt
from model import StationaryModelIP
from pymor.algorithms.pod import pod

#%% FOM IRGNM 

def IRGNM_FOM(IRGNM_setup, fom_IP, tol, q0):
    
    # read IRGNM_setup and opt_data
    display = IRGNM_setup['display_iteration_process']
    tau = IRGNM_setup["tau"]
    k_max = IRGNM_setup["k_max"]
    noise_level = fom_IP.opt_data["noise_level"]
    Theta = IRGNM_setup['Theta']
    theta = IRGNM_setup['theta']
    
    # initialize
    start_time = timer()
    q = q0.copy()
    fom_IP.check_parameter_bounds(q)
    u, p, A_q = fom_IP.solve_and_assemble(q)                                    # u, p, A_q
    start_time_B = timer()
    B_u = fom_IP.assemble_B_u(u)  
    end_time_B = timer() - start_time_B                                  
    output_res = fom_IP.nonlin_output_res(u)                                    # nonlin output residual, I_4=output_res**2
    k = 0                                                                       # init iteration counter
    alpha = IRGNM_setup['alpha0']
    history = {'output_res': [output_res], 'Alpha': [alpha], 
               'number_fom_solves': 2, 
               'regularized_gradient_norm': [],
               'non_regularized_gradient_norm': [],
               'number_fom_B_operator_applications': 0,
               'time_steps': [0],
               'cgiter': 0,
               'time_assemb_B_u': [end_time_B] ,
               }                                                        
    stagnation_flag = False
    if display == 1:
        print('===============================================================')
        print('Starting FOM IRGNM')
        print(f' with termination tolerance {tol+tau*noise_level} and noise level {noise_level}')
    
    #=========== LOOP ================================
    while output_res >= tol+tau*noise_level and k<k_max:                        # discrepancy principle
        
        #=========== get regularization parameter ================================
        fom_solves_k = 0
        regularization_qualification = False
        count = 1  
        Alpha = [alpha]
        #=========== CG ================================
        d, hist_inner =  solve_lin_quad_ocp_with_cg(IRGNM_setup,fom_IP,            
                                                 u,p,q,A_q,B_u,alpha)      # solve lin quad ocp with cg
        history['number_fom_solves']+= hist_inner['i']*2+1
        history['number_fom_B_operator_applications'] += hist_inner['i']*2+1
        history['cgiter'] += hist_inner['i']
        fom_solves_k += hist_inner['i']*2+1
        u_q_d = fom_IP.lin_state(q,d, u = u, A_q = A_q , B_u = B_u)
        lin_res = fom_IP.lin_output_res(u, u_q_d, None, None)     
        condition_low = 0.5*theta*output_res**2<0.5*lin_res**2
        condition_up = 0.5*lin_res**2 < 0.5*Theta*output_res**2
        regularization_qualification = condition_low and condition_up
        lin_res_old = 0
        
        while regularization_qualification == False or count < 50 :
            
            #### modify alpha
            if not condition_low:
                alpha *= 1.5  
            if not condition_up:
                alpha = max(alpha/2,1e-14)
            Alpha.append(alpha)
            #=========== CG ================================
            d, hist_inner =  solve_lin_quad_ocp_with_cg(IRGNM_setup,fom_IP,            
                                                     u,p,q,A_q,B_u,alpha)       # solve lin quad ocp with cg
            history['number_fom_solves']+= hist_inner['i']*2+1
            history['number_fom_B_operator_applications'] += hist_inner['i']*2+1
            # history['gradients'].append(hist_inner['rhs_cg'])
            history['cgiter'] += hist_inner['i']
            fom_solves_k += hist_inner['i']*2+1
            u_q_d = fom_IP.lin_state(q,d, u = u, A_q = A_q , B_u = B_u)
            lin_res = fom_IP.lin_output_res(u, u_q_d, None, None)
            
            #### check condition
            condition_low = 0.5*theta*output_res**2<0.5*lin_res**2
            condition_up = 0.5*lin_res**2< 0.5*Theta*output_res**2
            regularization_qualification = condition_low and condition_up
            count += 1
            if display == 1 and 0:
                print('        Search for regularization alpha:')
                print(f'        Try {count}: {0.5*theta*output_res**2:3.4e}< {0.5*lin_res**2:3.4e}<{0.5*Theta*output_res**2:3.4e} ?, alpha = {alpha:3.4e}')
            if regularization_qualification == True:
                regularization_choice_flag =f'              Regularization search succesfull after {count} tries'
                break
            if count > 2:
                if lin_res_old -lin_res < 1e-14 or Alpha[-1] == Alpha[-2] == 1e-14 :
                    regularization_choice_flag = f'              Regularization search stopped after {count} tries'
                    break
            lin_res_old = lin_res
        
        #=========== display =================================================
        if display == 1:
            #print(regularization_choice_flag)
            inner_flag = f'inner iter {count}, total cg fom solves {fom_solves_k} '
            print(f'k = {k:<3}: output_res = {output_res:3.4e},'\
                  f' alpha = {alpha:1.4e}, ' +inner_flag)
                  # + hist_inner['exit_flag'])     
        #=========== UPDATE ===================================================
        q += d
        fom_IP.check_parameter_bounds(q)
        u, p, A_q = fom_IP.solve_and_assemble(q)                                # u, p, A_q
        output_res = fom_IP.nonlin_output_res(u)

        # update history
        # TODO update history function
        reg_grad_obd = fom_IP.Q_prod_py.apply_inverse(fom_IP.Q_vector_space.from_numpy(hist_inner['rhs_cg'])).to_numpy()[0]
        non_reg_grad_obd = fom_IP.Q_prod_py.apply_inverse(fom_IP.Q_vector_space.from_numpy(hist_inner['non_regularized_gradient'])).to_numpy()[0]
        history['regularized_gradient_norm'].append(np.sqrt(reg_grad_obd.T@hist_inner['rhs_cg']))
        history['non_regularized_gradient_norm'].append(np.sqrt(non_reg_grad_obd.T@hist_inner['non_regularized_gradient']))
        history['output_res'].append(output_res)
        history['Alpha'].append(alpha)
        history['number_fom_solves']+= 2
        # history['iterates'].append(q.copy())
        
        # stagnation check
        if k > 3:
           if abs(history['output_res'][-1] -  history['output_res'][-2]) <1e-13 and abs(history['output_res'][-3] -  history['output_res'][-2]) <1e-13:
                stagnation_flag = True
                break
            
        B_u = fom_IP.assemble_B_u(u)  
        history['time_steps'].append(timer() - start_time)
        k += 1
    
    #=========== save and print history =======================================
    rhs_cg, lhs_cg, B_u_T_p = fom_IP.prepare_nullspace_CG(A_q,B_u,p,q,alpha) 
    reg_grad_obd = fom_IP.Q_prod_py.apply_inverse(fom_IP.Q_vector_space.from_numpy(rhs_cg)).to_numpy()[0]
    non_reg_grad_obd = fom_IP.Q_prod_py.apply_inverse(fom_IP.Q_vector_space.from_numpy(B_u_T_p)).to_numpy()[0]
   # history['gradients'].append(rhs_cg)
    history['regularized_gradient_norm'].append(np.sqrt(rhs_cg.T@non_reg_grad_obd))
    history['non_regularized_gradient_norm'].append(np.sqrt(B_u_T_p.T@non_reg_grad_obd))
    history['time'] = timer() - start_time
    history['k'] = k
    history['q'] = q
    if k == k_max:
        history['flag'] = f'FOM IRGNM reached maxit k = {str(k)}'\
                          f' with output_res = {output_res:3.4e} and total cg iter of {history["cgiter"]}'
    elif  k < k_max and not  stagnation_flag:
        history['flag'] = f'FOM IRGNM converged at k = {str(k)}'\
                          f' with output_res = {output_res:3.4e} and total cg iter of {history["cgiter"]}'
    elif stagnation_flag:
        history['flag'] = f'FOM IRGNM stagnated at k = {str(k)}' \
                          f' with output_res = {output_res:3.4e} and total cg iter of {history["cgiter"]}'
    print('------------------------ RESULTS ---------------------------------')
    print(history['flag'])
    if display == 1:
        print(f'Elapsed time is {history["time"]:4.5f} seconds.')
        print(f'Total FOM solves {history["number_fom_solves"]}.')
    print('------------------------------------------------------------------')
        #print(f'Regularization parameters {history["Alpha"]}.')
    return q, history

#%% Qr IRGNM

def Qr_IRGNM(IRGNM_setup, fom_IP, tol, q0, initial_Qsnapshots = [], k_max = None):
    
    # setup
    solver_option = 'direct'
    snapshot_invert = True
    tau = IRGNM_setup["tau"]
    if k_max is None:
        k_max = IRGNM_setup["k_max"]
    noise_level = fom_IP.opt_data["noise_level"]
    inner_SETUP = IRGNM_setup["Inner_IRGNM_setupQr"]
    inner_SETUP['display_iteration_process'] = False
    save = False
    plot_and_save = False
    if 1:
        print('===============================================================')
        print('Starting Qr IRGNM')
        print(f' with termination tolerance {tol+tau*noise_level} and noise level {noise_level}')

     # init   
    start_time = timer()
    k = 0
    q = q0.copy()
    fom_IP.check_parameter_bounds(q)
    u, p, A_q = fom_IP.solve_and_assemble(q, solver_option = solver_option)     # u, p, A_q
    start_time_B = timer()
    B_u = fom_IP.assemble_B_u(u)  
    time_assemb_B_u = timer() - start_time_B                                    # assemble B_u
    output_res = fom_IP.nonlin_output_res(u)
    alpha0 = IRGNM_setup['alpha0']
    history = {'output_res': [output_res],
               'number_fom_solves': 2, 'time1e-5':  0, 
               'time_assemb_B_u': [time_assemb_B_u],
               'time_steps': [0]}
    
    # Parameter Space Enrichment
    q_snapshots = [q]
    if not fom_IP.Q_norm(q-fom_IP.opt_data['q_0'])< 1e-13:
        q_snapshots.append(fom_IP.opt_data['q_0'])
    if 1:
        gradient = fom_IP.gradient(B_u, p, q, tikonov_parameter = None)         # compute gradient
        history['number_fom_B_operator_applications'] = 1
        if snapshot_invert and 1:
           gradient_obd = fom_IP.Q_prod_py.apply_inverse(fom_IP.Q_vector_space.from_numpy(gradient)).to_numpy()[0]
           q_snapshots.append(gradient_obd)
           history['number_fom_solves'] += 1
           grad_Q_norm = np.sqrt(gradient_obd.T@gradient)
        else: 
           q_snapshots.append(gradient)
           grad_Q_norm = fom_IP.Q_norm(gradient)  
    else: # Possibility 2
        alpha = 0                                                      
        gradient, lhs_cg, B_u_T_p = fom_IP.prepare_nullspace_CG(A_q,B_u,p,q,alpha)        
        d = q                                                    
        hessian_action,_,_ = lhs_cg(d)    
        q_snapshots = [q, gradient, hessian_action]
        history['number_fom_B_operator_applications'] = 3

    # initialize q reduced
    basis_Q = fom_IP.Q_vector_space.empty()
    Qfom_IP, basis_Q, basis_offset, old_basis_len, coarse_flag, _ = update_Q_space(
        fom_IP, None, initial_Qsnapshots + q_snapshots, basis_Q, k)
    
    # project q in basis
    q_r = basis_Q.to_numpy().dot(fom_IP.prod_matrix.dot(q))
    history['iterates_projected'] = [q_r.copy()]
    history['non_regularized_gradient_norm'] = [grad_Q_norm]
    
    # plot
    if plot_and_save:
        QQQ = basis_Q.to_numpy()
        for ind in range(len(basis_Q)):
            plot_and_save_iterate(QQQ[ind,:], f'basis {ind}', save)
        plot_and_save_iterate(q, f'iterate {0}', save)  
    qold = np.zeros_like(q)
    
    while output_res >= tol+tau*noise_level and k<k_max:
            
        print(f'k = {k:<3}, len Q_basis = {len(basis_Q)}, output_res = {output_res:3.4e}, '
              f'norm_grad = {history["non_regularized_gradient_norm"][-1]:3.4e}')

        ##### solve with IRGNM
        q_r, history_IRGNM = innerIRGNM(inner_SETUP, Qfom_IP, tol, q_r, alpha0 = alpha0, u = u , p = p, A_q = A_q, B_u = None)
        alpha0 = history_IRGNM['Alpha'][1]
        q = qold + basis_Q.lincomb(q_r).to_numpy()[0]                          # reconstruct snapshot
        fom_IP.check_parameter_bounds(q)
        u, p, A_q = fom_IP.solve_and_assemble(q, solver_option = solver_option) # u, p, A_q
        start_time_B = timer()
        B_u = fom_IP.assemble_B_u(u)  
        history['time_assemb_B_u'].append( timer() - start_time_B )  
        output_res = fom_IP.nonlin_output_res(u)   
        
        # Possibility 1
        if 1:
            gradient = fom_IP.gradient(B_u, p, q, tikonov_parameter = None)     # compute gradient
            history['number_fom_B_operator_applications'] += 1
            q_snapshots = []
            if snapshot_invert and 1:
               gradient_obd = fom_IP.Q_prod_py.apply_inverse(fom_IP.Q_vector_space.from_numpy(gradient)).to_numpy()[0]
               q_snapshots.append(gradient_obd)
               grad_Q_norm = np.sqrt(gradient_obd.T@gradient)
               history['number_fom_solves'] += 1
            else: 
               q_snapshots.append(gradient)
               grad_Q_norm = fom_IP.Q_norm(gradient)
        else:
            maxit = 1 
            d, hist_cg = conjugate_gradient(lhs_cg,gradient, q,
                                        maxit, tol =  IRGNM_setup["cg_tol"])
            q_snapshots = [gradient, d]
    
        # update Q space
        Qfom_IP, basis_Q, basis_offset, old_basis_len, coarse_flag, new_q_scale = update_Q_space(fom_IP, Qfom_IP, q_snapshots,
                                                                                 basis_Q, k,
                                                                                 other_snapshots = [q, q0.copy()])
        if new_q_scale:
            q_r = np.array([0.])
            qold = q
        else:
            q_r = basis_Q.to_numpy().dot(fom_IP.prod_matrix.dot(q))
        
        #### update history and plot
        k += 1
        history['non_regularized_gradient_norm'].append(grad_Q_norm)
        history['output_res'].append(output_res)
        # history['iterates_projected'].append(q_r.copy())
        # history['iterates_full'].append(q.copy())
        history['number_fom_solves'] += 2 + history_IRGNM['number_fom_solves']
        history['time_steps'].append(timer() - start_time)
            
        # plot
        if plot_and_save:
            QQQ = basis_Q.to_numpy()
            if coarse_flag:
               for ind in range(len(basis_Q)):
                   ind = old_basis_len + ind
                   plot_and_save_iterate(QQQ[old_basis_len + ind,:], f'basis {old_basis_len + ind}', save)
            else:   
                for ind in range(basis_offset):
                    indd = old_basis_len + ind
                    plot_and_save_iterate(QQQ[old_basis_len + indd,:], f'basis {old_basis_len + ind}', save)
            plot_and_save_iterate(q, f'iterate {k}', save)
    
    # finalize history
    history['time'] = timer() - start_time
    history['k'] = k
    history['len_Q_basis'] = len(basis_Q)
    history['q'] = q
    if k == k_max:
        history['flag'] = f'Qr IRGNM reached maxit k = {str(k)}'\
                          f' with output_res = {output_res:3.4e}'
    elif  k < k_max:
        history['flag'] = f'Qr IRGNM converged at k = {str(k)}' \
                          f' with output_res = {output_res:3.4e}'
    print('------------------------ RESULTS ---------------------------------')
    print(history['flag'])
    # print(f'Time to assemble B_u: {sum(history["time_assemb_B_u"])}')
    # print(f'Elapsed time to reach 1e-5 is {history["time1e-5"]:4.5f} seconds.')
    print(f'Elapsed time is {history["time"]:4.5f} seconds.')
    print(f'Total FOM solves {history["number_fom_solves"]}.')
    print(f'Basis size Q {len(basis_Q)}.')
    print('------------------------------------------------------------------')
    return q, q_r , history, Qfom_IP, basis_Q

########################################################################
def update_Q_space(fom_IP, Qfom_IP, new_snapshots_Q, basis_Q, k, other_snapshots = None, current_ROM_iterate = None):
    
    # BUILD RB: orthonormalize snapshots
    Q_PRODUCT = fom_IP.Q_prod_py 
    coarse_flag = False
    new_q_scale = False
    basis_len_before_enrichment = len(basis_Q)
    coarsening_Q = False
    coarse_iter = 10
    if k % coarse_iter == 0 and k != 0 and coarsening_Q:
        assert 0, 'Disable at own risk!'
        coarse_flag = True
        new_q_scale = False
        if 0: 
           for q_snapshot in new_snapshots_Q:
               q_space = fom_IP.Q_vector_space.from_numpy(q_snapshot)
               basis_Q.append(q_space, remove_from_other = (not True))
           basis_Q, singular_values = pod(basis_Q, modes = 5)
           basis_Q = gram_schmidt(basis_Q, product = Q_PRODUCT, copy = False)
           plt.figure()
           plt.semilogy(singular_values)
           plt.show()
           print(f'POD: new basis length is {len(basis_Q)}!')

        elif new_q_scale:
            new_constant_operator = fom_IP.constant_operator + fom_IP.true_parameterized_operator.assemble(fom_IP.parameters.parse(other_snapshots[0]))
            constant_operator = new_constant_operator.assemble()
            basis_Q = fom_IP.Q_vector_space.empty()
            basis_len_before_enrichment = 0
            for q_snapshot in new_snapshots_Q:
                q_in_Q_space = fom_IP.Q_vector_space.from_numpy(q_snapshot)
                try:
                    basis_length = len(basis_Q)
                    basis_Q.append(q_in_Q_space, remove_from_other = (not True))
                    basis_Q, R = gram_schmidt(basis_Q, product = Q_PRODUCT, return_R=True, atol=1e-13, rtol=1e-13,
                                              offset=basis_length, copy=False, check=False)
                except:
                    pass

        else: # reset basis completely
            assert other_snapshots is not None, 'If you want to coarse the Q basis please insert the current iterate also into the basis ....'
            constant_operator = fom_IP.constant_operator
            # old_basis_Q = basis_Q
            basis_Q = fom_IP.Q_vector_space.empty()
            basis_len_before_enrichment = len(basis_Q)
            new_snapshots_Q += other_snapshots
            for q_snapshot in new_snapshots_Q:
                q_space = fom_IP.Q_vector_space.from_numpy(q_snapshot)
                try:
                    basis_length = len(basis_Q)
                    basis_Q.append(q_space, remove_from_other = (not True))
                    basis_Q, R = gram_schmidt(basis_Q, product = Q_PRODUCT, return_R=True, atol=1e-13, rtol=1e-13,
                                              offset=basis_length, copy=False, check=False)
                except:
                    pass
       
    else: #standard gram schmidt
        constant_operator = fom_IP.constant_operator
        for q_snapshot in new_snapshots_Q:
            q_space = fom_IP.Q_vector_space.from_numpy(q_snapshot)
            try:
                basis_length = len(basis_Q)
                basis_Q.append(q_space, remove_from_other = (not True)) 
                basis_Q, R = gram_schmidt(basis_Q, product = Q_PRODUCT, return_R=True, atol=1e-13,
                                          rtol=1e-13, offset=basis_length, copy=False, check=False)
            except:
                pass
            
    basis_len_after_enrichment = len(basis_Q)
    basis_offset = basis_len_after_enrichment - basis_len_before_enrichment 
     
    # Assemble or extend affine decomposition
    if Qfom_IP is not None and not coarse_flag and 1:                             # expand the operator by the new basis vectors, if not coarsed
        operator_list = list(Qfom_IP.operator.operators)
        # create new operator
        for i in range(basis_offset):
            index = i + basis_len_before_enrichment
            q_i = basis_Q[index].to_numpy()[0]
            A_q = fom_IP.assemble_A_q_classic( q_i)
            assembled_op = NumpyMatrixOperator(A_q, source_id = 'STATE', range_id = 'STATE')
            operator_list.append(assembled_op)
        operator_coeff =  [1]
        for i in range(len(basis_Q)):
            operator_coeff.append(ProjectionParameterFunctional('q_coeff', len(basis_Q), i))
    else:                                                                      # build oeprators from scratch
        operator_list = [constant_operator]                            
        operator_coeff = [1]
        for i in range(len(basis_Q)):
            q_i = basis_Q[i].to_numpy()[0]
            A_q = fom_IP.assemble_A_q_classic( q_i)
            assembled_op = NumpyMatrixOperator(A_q, source_id = 'STATE', range_id = 'STATE')
            operator_list.append(assembled_op)
            operator_coeff.append(ProjectionParameterFunctional('q_coeff', len(basis_Q), i))      

    # form lincomb operator
    q_operator = LincombOperator(operator_list, operator_coeff)
    
    # place operators into model
    new_primal_model  = fom_IP.primal_model.with_(operator = q_operator)
    new_opt_data = deepcopy(fom_IP.opt_data)
    new_opt_data['coercivity_q'] = lambda q_r: fom_IP.opt_data['coercivity_q'](basis_Q.lincomb(q_r['q_coeff']))
    if 'P1' in fom_IP.opt_data['problem_type']:
        new_opt_data['Q_prod_H1_matrix'] = project(fom_IP.products['h1'], basis_Q, basis_Q, product = Q_PRODUCT).matrix
        new_opt_data['Q_prod_L2_matrix'] = project(fom_IP.products['h1'], basis_Q, basis_Q, product = Q_PRODUCT).matrix    
    new_opt_data['bounds'] = [basis_Q.to_numpy().dot(fom_IP.opt_data['bounds'][0]), basis_Q.to_numpy().dot(fom_IP.opt_data['bounds'][1]) ]
    new_opt_data['par_dim'] = len(basis_Q)
    new_opt_data['q_0'] = basis_Q.to_numpy().dot(fom_IP.prod_matrix.dot(fom_IP.opt_data['q_0']))
    new_Qfom_IP = StationaryModelIP(primal_model = new_primal_model, primal_data = fom_IP.primal_data,
                                      opt_data = new_opt_data,
                                      boundary_info=fom_IP.boundary_info, name='Qfom_IP')

    return new_Qfom_IP, basis_Q, basis_offset, basis_len_before_enrichment, coarse_flag, new_q_scale


def innerIRGNM(IRGNM_setup, fom_IP, tol, q0, alpha0 = None,  u = None, p = None, A_q = None, B_u = None):
    
    # read IRGNM_setup and opt_data
    display = IRGNM_setup['display_iteration_process']
    tau = IRGNM_setup["tau"]
    k_max = IRGNM_setup["k_max"]
    noise_level = fom_IP.opt_data["noise_level"]
    Theta = IRGNM_setup['Theta']
    theta = IRGNM_setup['theta']
    
    # initialize
    start_time = timer()
    q = q0.copy()
    alpha = alpha0
    init_number_fom_solves = 0
    if u is None and p is None and A_q is None:
        u, p, A_q = fom_IP.solve_and_assemble(q)   
        init_number_fom_solves = 2                                  
    if B_u is None:
        B_u = fom_IP.assemble_B_u(u)                                              
        
    output_res = fom_IP.nonlin_output_res(u)                                    
    k = 0                                                                      
    #=========== get regularization parameter ================================
    if alpha is None:
       alpha = 1
    history = {'output_res': [output_res], 'Alpha': [alpha], 'number_fom_solves': init_number_fom_solves}
    history['gradients'] = []
    history['cgiter'] = 0
    if display == 1:
        print(f'Starting inner IRGNM with total tol = {tol+tau*noise_level} and noise_level {noise_level}')
    stagnation_flag = False
    #=========== LOOP ================================
    while output_res >= tol+tau*noise_level and k<k_max:
        
        #=========== get regularization parameter ================================
        regularization_qualification = False
        count = 1  
        lin_res_old = 0
        Alpha = [alpha]
        #=========== CG ================================
        d, hist_inner =  solve_lin_quad_ocp_with_cg(IRGNM_setup, fom_IP,
                                                    u, p, q, A_q, B_u, alpha)
        history['number_fom_solves']+= hist_inner['i']*2+1
        history['cgiter'] += hist_inner['i'] 
        u_q_d = fom_IP.lin_state(q,d, u = u, A_q = A_q , B_u = B_u)
        lin_res = fom_IP.lin_output_res(u, u_q_d, None, None)     
        condition_low = 0.5*theta*output_res**2<0.5*lin_res**2
        condition_up = 0.5*lin_res**2< 0.5*Theta*output_res**2
        regularization_qualification = condition_low and condition_up
        
        while regularization_qualification == False or count < 50 :

            if not condition_low:
                alpha *= 1.5  
            if not condition_up:
                alpha = max(alpha/2,1e-14)
            Alpha.append(alpha)
            #=========== CG ================================
            d, hist_inner =  solve_lin_quad_ocp_with_cg(IRGNM_setup,fom_IP,            
                                                     u,p,q,A_q,B_u,alpha)
            history['number_fom_solves']+= hist_inner['i']*2+1
            history['cgiter'] += hist_inner['i']
            history['gradients'].append(hist_inner['rhs_cg'])
            u_q_d = fom_IP.lin_state(q,d, u = u, A_q = A_q , B_u = B_u)
            lin_res = fom_IP.lin_output_res(u, u_q_d, None, None)
            
            #### check condition
            condition_low = 0.5*theta*output_res**2<0.5*lin_res**2
            condition_up = 0.5*lin_res**2< 0.5*Theta*output_res**2
            regularization_qualification = condition_low and condition_up
            if display == 1:
                print(f'        Try {count}: {0.5*theta*output_res**2:3.4e}< {0.5*lin_res**2:3.4e}<{0.5*Theta*output_res**2:3.4e} ?, alpha = {alpha:3.4e}')
            if regularization_qualification == True:
                regularization_choice_flag =f' Regularization search succesfull after {count} tries'
                break
            if count > 2:
                if lin_res_old -lin_res < 1e-14 or Alpha[-1] == Alpha[-2] == 1e-14 :
                    regularization_choice_flag = f' Regularization search stopped after {count} tries'
                    break
            lin_res_old = lin_res
            count += 1
        
        #=========== display =================================================
        if display == 1:
            print(f'k = {k:<3}, output_res = {output_res:3.4e},'\
                  f' alpha = {alpha:1.4e}, ' 
                  + hist_inner['exit_flag'] + regularization_choice_flag)
            
        #=========== UPDATE ===================================================
        q += d
        #fom_IP.check_parameter_bounds(q)
        u, p, A_q = fom_IP.solve_and_assemble(q)                                # u, p, A_q
        output_res = fom_IP.nonlin_output_res(u)
        # update history
        history['output_res'].append(output_res)
        history['Alpha'].append(alpha)
        history['number_fom_solves']+= 2
        # history['iterates'].append(q.copy())
        # stagnation check
        if k > 3:
           if abs(history['output_res'][-1] -  history['output_res'][-2]) <1e-12:
                stagnation_flag = True
                break     
        B_u = fom_IP.assemble_B_u(u)  
        k += 1
    #=========== save and print history =======================================
    rhs_cg, lhs_cg, B_u_T_p = fom_IP.prepare_nullspace_CG(A_q,B_u,p,q,alpha) 
    history['gradients'].append(rhs_cg)
    history['time'] = timer() - start_time
    history['k'] = k
    if k == k_max:
        history['flag'] = f'        Inner IRGNM reached maxit k = {str(k)}'\
                          f' with output_res = {output_res:3.4e}, time is {history["time"]:4.5f} with total cg iter of {history["cgiter"]}'
    elif  k < k_max and not  stagnation_flag:
        history['flag'] = f'        Inner IRGNM converged at k = {str(k)}'\
                          f' with output_res = {output_res:3.4e}, time is {history["time"]:4.5f} with total cg iter of {history["cgiter"]}'
    elif stagnation_flag:
        history['flag'] = f'        Inner IRGNM stagnated at k = {str(k)}' \
                          f' with output_res = {output_res:3.4e}, time is {history["time"]:4.5f} with total cg iter of {history["cgiter"]}'
    #print('------------------------------------------------------------------')
    print(history['flag'])
    if display == 1:
        print(f'Total FOM solves {history["number_fom_solves"]}.')
        print(f'Regularization parameters {history["Alpha"]}.')
    return q, history 

#%% Solving linear quadratic optimal control problem with cg

def solve_lin_quad_ocp_with_cg(IRGNM_setup,model_IP, u, p, q, A_q, B_u, alpha):
    
    # read cg set up
    cg_tol =  IRGNM_setup['cg_tol']
    cg_maxit = IRGNM_setup['cg_maxit']   
    par_dim = model_IP.opt_data['par_dim']

    # prepare cg lhs, rhs and initial guess  
    rhs_cg, lhs_cg, B_u_T_p = model_IP.prepare_nullspace_CG(A_q, B_u, p, q, alpha)
    d0 = np.zeros((par_dim,))
    
    d, hist_cg = conjugate_gradient(lhs_cg,rhs_cg, d0,
                                    maxit = cg_maxit, tol = cg_tol)
        
    hist_cg['rhs_cg'] = rhs_cg
    hist_cg['non_regularized_gradient'] = B_u_T_p
    return d, hist_cg
