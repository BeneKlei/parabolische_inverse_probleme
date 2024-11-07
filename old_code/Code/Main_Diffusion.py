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
# Description:
# This is the main file for the numerical experiment in section 4.3: the reconstruction of the diffusion coefficient.

import numpy as np
import matplotlib.pyplot as plt
from pymor.basic import *
from IRGNM import IRGNM_FOM, Qr_IRGNM
from Qr_Vr_TR_IRGNM import Qr_Vr_TR_IRGNM
from problems import whole_problem
from discretizer import discretize_stationary_IP
from pymor.parameters.base import ParameterSpace

#%% Set options    
    
# general options           
N = 300                                                                        # FE Dofs = (N+1)^2                                                
tol = 1e-14                                                                    # safeguard tolerance
tau = 3.5                                                                      # discrepancy parameter
k_max = 50                                                                     # maxit
noise_level = 1e-5                                                             # noise level  

# choose optimization methods
opt_methods = {     
              "Qr IRGNM",
              "FOM IRGNM",
              "Qr-Vr TR IRGNM",
               }

# choose norm in the parameter space
norm_type = 'H1'

# plot and save options
plot_parameters = True
save_plots = True
save_path = 'Diffusion_Plots/'

#%% Create analytical problem and discretize it

# define analytical PDE problem
print('Construct problem..')                                                     
analytical_problem, q_exact, N, problem_type, exact_analytical_problem, energy_problem = whole_problem(
                                                        N = N,
                                                        parameter_location = 'diffusion',
                                                        boundary_conditions = 'dirichlet',
                                                        exact_parameter = 'PacMan',
                                                       )

# define data for the inverse problem and store in opt_data
print('Discretizing problem...')                                                
diameter = np.sqrt(2)/N
grid_type = RectGrid
low = 0.001                                                                     # lower bound value
up = 1e20                                                                       # upper bound value 
par_dim = (N+1)**2                                                              # dimension of parameter space
q_low = low*np.ones((par_dim,))                                                 # lower bound
q_up =  up*np.ones((par_dim,))                                                  # upper bound
bounds = [q_low, q_up]
q0 = 3*np.ones((par_dim,))                                                      # initial guess for IRGNM
q_0 =  3*np.ones((par_dim,))                                                    # regularization center for IRGNM
opt_data = {'FE_dim': (N+1)**2,                                                 # FE dofs
            'par_dim': par_dim,                                                     
            'noise_level': noise_level,
            'q_exact': q_exact,
            'q0': q0,
            'q_0': q_0, 
            'low': low,
            'up': up,
            'noise_percentage': None,   
            'problem_type': problem_type,
            'q_in_h1': True,
            'norm_on_q': norm_type,
            'B_assemble': False,                                                # options to preassemble affine components or not
            'bounds': bounds
            } 

# create parameter space
parameter_space = ParameterSpace(analytical_problem.parameters, [low,up]) 

# discretize analytical problem to obtain inverse problem fom
fom_IP, fom_IP_data = discretize_stationary_IP(analytical_problem,
                                               diameter = diameter,
                                               opt_data = opt_data,
                                               exact_analytical_problem = exact_analytical_problem,
                                               energy_product_problem = energy_problem,
                                               grid_type = grid_type
                                               )

#%% Solver setup

cg_tol = 1e-13                                                                  # tolerance for the cg algorithm
cg_maxit = None                                                                 # maxiter for the cg algorithm
theta = .4                                                                      # lower parameter for the acceptance of regularization parameter alpha
Theta = .9                                                                      # upper parameter for the acceptance of regularization parameter alpha
relax_factor = 0                                                                # relax the error estimate
alpha0 = 1e-3                                                                   # initial regularization
reductor_type = 'simple_coercive'                                               # choose pyMOR reductor type

# FOM IRGNM setup
IRGNM_setup = {'display_iteration_process': 1,                                 
              'tau': tau,
              'tol': tol,
              'k_max': k_max,
              'bounds': bounds,
              'cg_tol': cg_tol,
              'cg_maxit': cg_maxit,
              'theta': theta,
              'Theta': Theta,                                                   
              'relax_factor': relax_factor,                                     
              'alpha0': alpha0
              }

# Setup for the inner IRGNM in Qr IRGNM
Inner_IRGNM_setupQr = {'display_iteration_process': 0,                         
                    'tau': 1,
                    'tol': tol,
                    'k_max': 2,
                    'noise_level': noise_level,    
                    'bounds': bounds,
                    'cg_tol': cg_tol,
                    'cg_maxit': cg_maxit,
                    'theta': theta,
                    'Theta': Theta, 
                    'alpha0': alpha0
                     }

IRGNM_setup['Inner_IRGNM_setupQr'] = Inner_IRGNM_setupQr

# Setup for the inner IRGNM in Qr-Vr TR IRGNM
Inner_IRGNM_setupQrVr = {'display_iteration_process': 1,                           
                    'tau': 1,
                    'tol': tol,
                    'k_max': 40,
                    'noise_level': noise_level,    
                    'bounds': bounds,
                    'cg_tol': cg_tol,
                    'cg_maxit': cg_maxit,
                    'theta': theta,
                    'Theta': Theta, 
                    'alpha0': alpha0
                     }

# Setup for Qr-Vr TR IRGNM
TR_IRGNM_setup = {'Inner_IRGNM_setup': Inner_IRGNM_setupQrVr,
                 'tau': tau,                                                    # discrepancy parameter
                 'noise_level': noise_level,                                    # moise level
                 'theta': theta,                                                # lower bound for the choice of the regularization
                 'Theta': Theta,                                                # upper bound for the choice of the regularization
                 'beta': .95,                                                   # trust region boundary parameter
                 'radius': 1,                                                   # init trust region radius
                 'shrink_factor': .5,                                
                 'miniter':0, 
                 'maxiter':k_max, 
                 'tol': tol,                                                    # additional tolerance for usage with nosie level 0
                 'radius_tol':.75,                                              # treshhold for increasing radius
                 'line_search_params': None, 
                 'stagnation_window': 3,
                 'stagnation_threshold': np.inf, 
                 'return_stages': False, 
                 'return_subproblem_stages': False,
                 'reductor_type': reductor_type,
                 'bounds': bounds,
                 'display_iteration_process': 1,
                 'outer_regularization': False,                                 # use outer regularization
                 'alpha_init': alpha0,                                          # initial alpha for first inner IRGNM or outer regularization
                 'relax_factor': relax_factor,
                 'alpha0': alpha0
                 }

#%% Solve the inverse problem

#========== Qr IRGNM ==========================================================s
if 'Qr IRGNM' in opt_methods:
    q_Qr, q_Qr_r, history_QFOM, Qfom_IP, basis_Q = Qr_IRGNM(IRGNM_setup, fom_IP, tol, q0)

#=========== FOM IRGNM ========================================================
if 'FOM IRGNM' in opt_methods:
    q_FOM, history_FOM = IRGNM_FOM(IRGNM_setup, fom_IP, tol, q0)
    
#=============== Qr-Vr TR IRGNM ===============================================
if 'Qr-Vr TR IRGNM' in opt_methods:
    q_Qr_Vr, history_QTRRB = Qr_Vr_TR_IRGNM(parameter_space, fom_IP, TR_IRGNM_setup, q0, tol)  

#%% Visualization

print('#######################################################################')
print('################## ALL RESULTS ########################################')
print(f'problemtype = {problem_type}, N = {N}, regularization_type = {norm_type}')
print(f'tau = {tau}, noise_level = {noise_level}.')
print('#######################################################################')
print('#######################################################################')

# error est usage plot
if 'q_Qr_Vr' in locals() and 0:
     plt.figure()
     plt.plot(history_QTRRB['fom_solves_for_unassembled_residual_error_est'], '.-', label = 'Online computation of the error estimator')
     plt.plot(history_QTRRB['fom_solves_for_assembling_residual_error_est'],'.-', label = 'Assembly of the error estimator')
     plt.legend()
     plt.xlabel("k")
     plt.ylabel("FOM solves")
     max_len = len(history_QTRRB['fom_solves_for_unassembled_residual_error_est'])
     plt.xticks(ticks = range(0,max_len,5), labels = range(0,max_len, 5))
     if save_plots:
         if save_path is not None:
             title = save_path + problem_type+f'_N={N}_errorest.png'
         else:
             title = problem_type+f'_N={N}_errorest.png'
         plt.savefig(title, bbox_inches='tight')
         
# plot exact parameter
if plot_parameters:
   fom_IP.plot_matplotlib(q_exact, 'Exact parameter', save_plots, problem_type+f'_N={N}_exact.png', save_path)
            
# plot reconstructed parameters and print results
if 'q_Qr' in locals():
    print(f'Qr IRGNM|| time: {history_QFOM["time"]}, FOM solves: {history_QFOM["number_fom_solves"]}, FOM B applications: {history_QFOM["number_fom_B_operator_applications"]}, iterations: {history_QFOM["k"]}, Basis Q: {history_QFOM["len_Q_basis"]}')
    rel_err_QFOM = fom_IP.Q_norm(q_exact-q_Qr)/fom_IP.Q_norm(q_exact)
    print(f'discrepancy residual = {history_QFOM["output_res"][-1]}, norm_grad = {history_QFOM["non_regularized_gradient_norm"][-1]}, rel. error to exact = {rel_err_QFOM}')
    print('-------------------------------------------------------------------')
    if plot_parameters:
        plt.figure()
        history_QFOM['u'] = fom_IP.state(q_Qr).to_numpy()[0]
        fom_IP.plot_matplotlib(q_Qr, r'$q^{Q_r}$', save_plots, problem_type+f'_N={N}_QFOM.png', save_path)
        fom_IP.plot_matplotlib(history_QFOM['u'], r'$u^{Q_r}$', save_plots, problem_type+f'_N={N}_stateQFOM.png', save_path)  
        
    if save_plots:
        if save_path is not None:
            title = save_path + problem_type+f'QFOMarray_N={N}.npy'
        else:
            title = problem_type+f'QFOMarray_N={N}.npy'
        np.save(title, history_QFOM)

if 'q_FOM' in locals():
    print(f'FOM IRGNM|| time: {history_FOM["time"]}, FOM solves: {history_FOM["number_fom_solves"]}, FOM B applications: {history_FOM["number_fom_B_operator_applications"]}, iterations: {history_FOM["k"]}')
    rel_err_FOM = fom_IP.Q_norm(q_exact-q_FOM)/fom_IP.Q_norm(q_exact)
    print(f'discrepancy residual = {history_FOM["output_res"][-1]}, norm_grad = {history_FOM["non_regularized_gradient_norm"][-1]}, rel. error to exact = {rel_err_FOM}')
    print('-------------------------------------------------------------------')
    if plot_parameters:
        plt.figure()
        fom_IP.plot_matplotlib(q_FOM, r'$q^{FOM}$', save_plots, problem_type+f'_N={N}_FOM.png', save_path)
        history_FOM['u'] = fom_IP.state(q_FOM).to_numpy()[0]
        fom_IP.plot_matplotlib(history_FOM['u'], r'$u^{FOM}$', save_plots, problem_type+f'_N={N}_stateFOM.png', save_path)
            
    if save_plots:
        if save_path is not None:
            title = save_path + problem_type+f'FOMarray_N={N}.npy'
        else:
            title = problem_type+f'FOMarray_N={N}.npy'
        np.save(title, history_FOM)

if 'q_Qr_Vr' in locals():
    print(f'Qr-Vr TR IRGNM|| time: {history_QTRRB["time"]}, FOM solves: {history_QTRRB["number_fom_solves"]}, fom solves for error est: {history_QTRRB["true_fom_solves_for_residual_error_est"]}, FOM B applications: {history_QTRRB["number_fom_B_operator_applications"]}, iterations: {history_QTRRB["k"]}, Basis Q = {history_QTRRB["len_Q_basis"]}, Basis V = {history_QTRRB["len_V_basis"]}')
    rel_err_QTRRB = fom_IP.Q_norm(q_exact-q_Qr_Vr)/fom_IP.Q_norm(q_exact)
    print(f'discrepancy residual = {history_QTRRB["output_res"][-1]}, norm_grad = {history_QTRRB["non_regularized_gradient_norm"][-1]}, rel. error to exact = {rel_err_QTRRB}')
    print('-------------------------------------------------------------------')
    if plot_parameters:
        plt.figure()
        fom_IP.plot_matplotlib(q_Qr_Vr, r'$q^{Q_r-V_r}$', save_plots, problem_type+f'_N={N}_QTRRB.png', save_path)
        history_QTRRB['u'] = fom_IP.state(q_Qr_Vr).to_numpy()[0]
        fom_IP.plot_matplotlib(history_QTRRB['u'], r'$u^{Q_r-V_r}$', save_plots, problem_type+f'_N={N}_stateQTRRB.png', save_path)
        
    if save_plots:
        if save_path is not None:
            title = save_path + problem_type+f'TRRBarray_N={N}.npy'
        else:
            title = problem_type+f'TRRBarray_N={N}.npy'
        np.save(title, history_QTRRB)
            

# combined plots and plot differences    
if "FOM IRGNM" in opt_methods and 'Qr IRGNM' in opt_methods and 'Qr-Vr TR IRGNM' in opt_methods: 
    
    # subplots
    fom_IP.plot_subplot_para(q_exact, q_FOM, q_Qr, q_Qr_Vr, title = None, save = save_plots, save_title = problem_type+f'_N={N}_subplots_q.png', path = save_path)
    
    # H1 error
    print(f'rel. error norm((q_FOM - q_Qr)/q_FOM)= { fom_IP.Q_norm(q_FOM-q_Qr)/fom_IP.Q_norm(q_FOM)}')
    print(f'rel. error norm((q_Qr - q_Qr_Vr)/q_Qr) = { fom_IP.Q_norm(q_Qr-q_Qr_Vr)/fom_IP.Q_norm(q_Qr)}')
    print(f'rel. error norm((q_FOM - q_Qr_Vr)/q_FOM) = { fom_IP.Q_norm(q_FOM-q_Qr_Vr)/fom_IP.Q_norm(q_FOM)}')

    # L2 error
    q_FOM_pymor = fom_IP.Q_vector_space.from_numpy(q_FOM)
    q_Qr_pymor = fom_IP.Q_vector_space.from_numpy(q_Qr)
    q_Qr_Vr_pymor = fom_IP.Q_vector_space.from_numpy(q_Qr_Vr)
    q_FOM_norm = fom_IP.primal_model.l2_norm(q_FOM_pymor)
    print(f'rel. error L2 norm((q_FOM - q_Qr)/q_FOM)= {(fom_IP.primal_model.l2_norm(q_FOM_pymor-q_Qr_pymor)/q_FOM_norm)[0]}')
    print(f'rel. error L2 norm((q_Qr - q_Qr_Vr)/q_Qr) = { (fom_IP.primal_model.l2_norm(q_Qr_pymor-q_Qr_Vr_pymor)/fom_IP.primal_model.l2_norm(q_Qr_pymor))[0]}')
    print(f'rel. error L2 norm((q_FOM - q_Qr_Vr)/q_FOM) = { (fom_IP.primal_model.l2_norm(q_FOM_pymor-q_Qr_Vr_pymor)/q_FOM_norm)[0]}')
    
    # # maximum minimum pointwise error
    # print(f'max rel. error (q_FOM - q_Qr)/q_FOM)= { max(abs((q_FOM-q_Qr))/abs(q_FOM))}')
    # print(f'max rel. error (q_Qr - q_Qr_Vr)/q_Qr) = {max(abs((q_Qr-q_Qr_Vr))/abs(q_Qr))}')
    # print(f'max rel. error (q_FOM - q_Qr_Vr)/q_FOM) = { max(abs((q_FOM-q_Qr_Vr))/abs(q_FOM))}')
    # print(f'min rel. error (q_FOM - q_Qr)/q_FOM)= { min(abs((q_FOM-q_Qr))/abs(q_FOM))}')
    # print(f'min rel. error (q_Qr - q_Qr_Vr)/q_Qr) = {min(abs((q_Qr-q_Qr_Vr))/abs(q_Qr))}')
    # print(f'min rel. error (q_FOM - q_Qr_Vr)/q_FOM) = { min(abs((q_FOM-q_Qr_Vr))/abs(q_FOM))}')
    
    # # relative differences pointwise
    # plt.figure()
    # fom_IP.plot_matplotlib(abs((q_FOM-q_Qr))/abs(q_FOM), r'$d^{Q_r}$', save_plots, problem_type+f'_N={N}_rel_diff2_fom-qfom.png' , save_path )
    # plt.figure()
    # fom_IP.plot_matplotlib(abs((q_Qr-q_Qr_Vr))/abs(q_Qr), r'$d^{Q_r,Q_r-V_r}$', save_plots, problem_type+f'_N={N}_rel_diff2_qfom-qtrrb.png', save_path  )
    # plt.figure()
    # fom_IP.plot_matplotlib(abs((q_FOM-q_Qr_Vr))/abs(q_FOM), r'$d^{Q_r-V_r}$', save_plots, problem_type+f'_N={N}_rel_diff2_fom-qtrrb.png', save_path  )
    
# plot output residual against iterations
plt.figure()
if 'q_Qr' in locals():
    plt.semilogy(history_QFOM["output_res"], '-o', label= r'$Q_r$ IRGNM')
if 'q_Qr_Vr' in locals():
    plt.semilogy(history_QTRRB["output_res"], '-o', label= r'$Q_r$-$V_r$ IRGNM')
if 'q_FOM' in locals():
    plt.semilogy(history_FOM["output_res"], '-o', label= 'FOM IRGNM')
value = tol+noise_level*tau
plt.axhline(y=value, color='g', linestyle='-', label = r'$\tau\delta$')
plt.xlabel('$k$')
plt.ylabel(r'$\|\|F(q^k)-y^{\delta}\|\|_{H}$')
plt.legend(loc = 1)
if save_plots:
    if save_path is not None:
        title = save_path + problem_type+f'_N={N}_residual_iteration.png'
    else:
        title = problem_type+f'_N={N}_residual_iteration.png'
    plt.savefig(title, bbox_inches='tight')

# plot output residual against cpu time
plt.figure()
if 'q_Qr' in locals():
    plt.semilogy(history_QFOM["time_steps"], history_QFOM["output_res"], '-o', label= r'$Q_r$ IRGNM')
if 'q_Qr_Vr' in locals():
    plt.semilogy(history_QTRRB["time_steps"], history_QTRRB["output_res"], '-o', label= r'$Q_r$-$V_r$ IRGNM')
if 'q_FOM' in locals():
    plt.semilogy(history_FOM["time_steps"], history_FOM["output_res"], '-o', label= 'FOM IRGNM')
value = tol+noise_level*tau
plt.axhline(y=value, color='g', linestyle='-', label = r'$\tau\delta$')
plt.xlabel('time [s]')
plt.ylabel(r'$\|\|F(q^k)-y^{\delta}\|\|_{H}$')
plt.legend(loc = 1)
if save_plots:
    if save_path is not None:
        title = save_path + problem_type+f'_N={N}_residual_cpu.png'
    else:
        title = problem_type+f'_N={N}_residual_cpu.png'
    plt.savefig(title, bbox_inches='tight')

# # plot gradient norm against iterations
# plt.figure()
# if 'q_Qr' in locals():
#     plt.semilogy(history_QFOM["non_regularized_gradient_norm"], '-o', label= r'$Q_r$ IRGNM')
# if 'q_Qr_Vr' in locals():
#     plt.semilogy(history_QTRRB["non_regularized_gradient_norm"],'-o', label= r'$Q_r$-$V_r$ IRGNM')
# if 'q_FOM' in locals():
#     plt.semilogy(history_FOM["regularized_gradient_norm"], '-o', label= 'FOM IRGNM nonregularized')
# plt.xlabel('$k$')
# plt.legend(loc = 1)
# plt.title("Gradient norm")
# if save_plots:
#     if save_path is not None:
#         title = save_path + problem_type+f'_N={N}_gradient.png'
#     else:
#         title = problem_type+f'_N={N}_gradient.png'
#     plt.savefig(title, bbox_inches='tight')
