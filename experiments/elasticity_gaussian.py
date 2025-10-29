import numpy as np
import copy

import RBInvParam.problems.elasticity.material_model as mm

from RBInvParam.error_estimators.state_error_estimators import StateErrorEstimatorType
from RBInvParam.error_estimators.adjoint_error_estimators import AdjointErrorEstimatorType
from RBInvParam.error_estimators.objective_error_estimators import ObjectiveErrorEstimatorType


y_res = 30
z_res = 30
par_dim = (y_res + 1) * (z_res + 1) 
T_initial = 0
T_final = 5.0
nt = 50

delta_t = (T_final - T_initial) / nt

assert T_final > T_initial
q_circ = np.ones((1, par_dim))
q_exact = np.ones((1,par_dim))

q_exact[0,200] = 2
q_exact[0,300] = 3

q_circ[0,:] = 1.0

bounds = np.zeros((par_dim, 2))
bounds[:,0] = 1e-20
bounds[:,1] = 1e20

setup = {
    'spatial_resolution' : [4,y_res,z_res],
    'body_force' : {
        # 'type' : mm.BodyForceType.CenterExcite,
        # 'hyperparameter' : {}
        # 'type' : mm.BodyForceType.CenterExcite,
        # 'hyperparameter' : {
        #     'center': [-0.1,0,0],
        #     'sigma' : 1.0,
        # }
        'type' : mm.BodyForceType.Gaussian,
        'hyperparameter' : {
            'center': [-0.1,0.0,0.0],
            'sigma' : 1.0,
        }
    },
    'system_matrix' : {
        'type' : mm.SystemMatrixType.CosseratDelamination,
        'hyperparameter' : {
            'lambda' : 1e1,
            'mu' : 1e1,
            'nu' : 1e-3,
        }
    },
    'observation_operator': {
        'type': mm.ObservationOperatorType.SensorsR28d,                       # Type of observation operator (e.g., identity = full state observed)
        'hyperparameter' : {}
    },
    'dims' : {
        'nt': nt,                                     # Number of time steps
        'par_dim' : None,
        'state_dim' : None,
        'observation_space_dim': None
    },
    'products': {                                 # Inner products used in the problem
        'prod_H': 'l2',                           # Product on H_h
        'prod_Q': 'euclid',                       # Product on Q_h
        'prod_V': 'h1_0_semi',                    # Product on V_h
        'prod_C': 'euclid',                       # Product on C_h
    },
    'T_initial': T_initial,                       # Start time of the simulation
    'T_final': T_final,                           # End time of the simulation
    'delta_t': delta_t,                           # Time step size
    'noise_percentage': None,                     # Relative noise level, will be set by 'build_InstationaryModelIP'
    'noise_level': 5 * 1e-5,                      # Absolute noise magnitude added to data
    'q_circ': q_circ,                             # Backgroundlevel for the parameter
    'q_exact_function': None,                     # Exact parameter as function, will be set by 'build_InstationaryModelIP'
    'q_exact': q_exact,                           # Exact parameter values, will be set by 'build_InstationaryModelIP'
    'q_time_dep': False,                          # Whether parameter is time-dependent (bool)
    'riesz_rep_grad': True,                       # Use Riesz representative for gradient in optimization
    'bounds': bounds,                             # Bounds on parameter values (e.g., for optimization)
    'save_path' : None,
    'time_stepper' : {
        'name' : 'newman_second_order',
        'zeta' : 0.5
    }
}

q_start = q_circ

FOM_optimizer_parameter = {
    'method' : 'FOM_IRGNM',
    'q_0': q_start,                                          # Initial guess for the parameter to be optimized
    'alpha_0': 1e-5,                                          # Initial regularization parameter
    'tol': 1e-9,                                            # Absolute convergence tolerance for optimization
    'tau': 1.00,                                              # Relative (to the noise) convergence tolerance for optimization
    'noise_level': setup['noise_level'],                     # Noise level in observed data (from model setup)
    'theta': 0.4,                                         # Lower tolerance for the direction acceptance condition
    'Theta': 1.95,                                           # Upper tolerance for the direction acceptance condition
    #####################
    'i_max': 250,                                             # Maximum number of outer optimization iterations
    'reg_loop_max': 25,                                      # Maximum number of regularization updates per step
    'i_max_inner': 10,                                       # Maximum number of inner iterations
    ####################
    'lin_solver_parms': {
        'method' : 'gd',                                     # Method for solving linear systems (e.g., gradient descent)
        'max_iter': 1e3,                                     # Max iterations for the linear solver
        'lin_solver_tol': 1e-12,                          # Tolerance for convergence in the linear solver
        'kappa_arm' : 1e-12,
        'armijo_inital_step_size': 1,                                    # Initial step size for iterative linear solver
        'armijo_min_step_size' : 1e-20
    },
    'use_cached_operators': True ,                          # Whether to reuse assembled operators (improves speed if True)
    'dump_every_nth_loop': 1,                                # Dump intermediate results every n optimization iterations
}


TR_optimizer_parameter = {
    'method' : 'TR_IRGNM',
    'q_0': q_start,                                              # Initial guess for the parameter to be optimized        
    'alpha_0': 1e-5,                                              # Initial regularization parameter (data fidelity vs. regularization)        
    'tol': 1e-9,                                                 # Absolute convergence tolerance for optimization
    'tau': 1.00,                                                  # Relative (to the noise) convergence tolerance for optimization
    'noise_level': setup['noise_level'],                         # Noise level in observed data (from model setup)
    'theta': 0.4,
    'Theta': 1.95,                                               # Upper bound for step acceptance condition
    'tau_tilde': 3.5,                                            # Relative (to the noise) convergence tolerance for optimization inside the trust region
    'NCD' : False,
    'offline_parallel' : True,
    #####################
    'i_max': 75,                                                 # Max number of outer optimization iterations
    'reg_loop_max': 10,                                          # Max number of regularization updates per iteration
    'i_max_inner': 30,                                           # Max number of inner iterations
    'agc_armijo_max_iter': 50,                                  # Max iterations for computing the AGC
    'TR_armijo_max_iter': 10,                                     # Max iterations Armijo condition to enforce the trust-region 
    #####################
    'lin_solver_parms': {
        'method': 'gd',                                          # Method for solving linear systems (e.g., gradient descent)
        'max_iter': 1e3,                                         # Maximum iterations for the linear solver
        'lin_solver_tol': 1e-12,                                 # Convergence tolerance for the linear solver
        'kappa_arm' : 1e-12,
        'armijo_inital_step_size': 1,                                    # Initial step size for iterative linear solver
        'armijo_min_step_size' : 1e-20
    },
    # 'lin_solver_parms': {
    #     'method': 'BiCGSTAB',                                  # BiCGSTAB method for solving nonsymmetric linear systems
    #     'rtol': 1e-12,                                         # Relative convergence tolerance
    #     'atol': 1e-12,                                         # Absolute convergence tolerance
    #     'maxiter': 1e3                                         # Max iterations for BiCGSTAB solver
    # },
    'enrichment': {
        'parameter_basis' : {
            'reduced_basis' : True,
            'include_each_time_step' : False,
            'include_lin_grad' : False,
            'sample_every_n_th' : None,
            # 'normalize' : True,
            # 'HaPOD' : {
            #     'HaPOD_tol': 1e-1,    
            # },
            'normalize' : None,
            'HaPOD' : None,
            'overwrite_every_n' : False,
            'keep_last_n': None
        },
        'state_basis' : {
            'include_lins' : False,
            'sample_every_n_th' : None,
            'normalize' : True,
            'HaPOD' : {
                'HaPOD_tol': 1e-6,    
            },
            'overwrite_every_n' : False,
            'keep_last_n': None
        },
        'adjoint_basis' : {
            'sample_every_n_th' : None,
            'normalize' : False,
            'HaPOD' : {
                'HaPOD_tol': 1e-16,    
            },
            'overwrite' : False
        }
    },
    'error_estimator_types' : {
        'state' : StateErrorEstimatorType.NONE,
        'adjoint' : AdjointErrorEstimatorType.NONE,
        'objective' : ObjectiveErrorEstimatorType.NONE,
    },
    #####################
    'use_cached_operators': True,                               # Reuse previously assembled operators to save computation
    'dump_every_nth_loop': 1,                                    # Dump intermediate results every n optimization iterations
    #####################
    'eta0': 1e-2,                                                # Initial trust region tolerance
    'kappa_arm': 1e-12,                                          # Armijo condition constant for sufficient decrease
    'eta_min' : 1e-5,
    'eta_max' : 0.15,
    'beta_1': 0.95,                                              # Trust region edge tolerance.
    'beta_2': 3/4,                                               # Tolerance for the trustworthiness. 
    'beta_3': 0.5                                                # Shrinking/Enlarging factor for the trust region.
}

EXPERIMENTS = {}

##########################################################################################

sigmas = [1.0, 2.0, 5.0]
for sigma in sigmas:
    setup_width = copy.deepcopy(setup)
    setup_width['body_force']['hyperparameter']['sigma'] = sigma

    setup_sensors = copy.deepcopy(setup_width)
    setup_identity = copy.deepcopy(setup_width)

    setup_identity['observation_operator']['type'] = mm.ObservationOperatorType.Identity


    FOM_optimizer_parameter_ = copy.deepcopy(FOM_optimizer_parameter)
    TR_optimizer_parameter_ = copy.deepcopy(TR_optimizer_parameter)

    # --------------------------------------------------------------------------
    EXPERIMENTS[f'{sigma}_FOM_sensors'] = (setup_sensors, FOM_optimizer_parameter_)
    EXPERIMENTS[f'{sigma}_FOM_identity'] = (setup_identity, FOM_optimizer_parameter_)
    # --------------------------------------------------------------------------
    TR_optimizer_parameter__ = copy.deepcopy(TR_optimizer_parameter_)
    TR_optimizer_parameter__['enrichment']['state_basis']['HaPOD']['HaPOD_tol'] = 1e-3
    EXPERIMENTS[f'{sigma}_TR_sensors'] = (setup_sensors, TR_optimizer_parameter__)
    EXPERIMENTS[f'{sigma}_TR_identity'] = (setup_identity, TR_optimizer_parameter__)


prefix = 'gaussian'
EXPERIMENTS = {f"{prefix}_{k}": v for k, v in EXPERIMENTS.items()}