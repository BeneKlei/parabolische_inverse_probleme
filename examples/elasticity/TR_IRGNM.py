import numpy as np
import logging
from pathlib import Path
from datetime import datetime

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

from pymor.basic import *

import RBInvParam.problems.elasticity.material_model as mm

from RBInvParam.optimizer import QrVrROMOptimizer
from RBInvParam.utils.io import save_dict_to_pkl
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.problems.elasticity.build import build_InstationaryModelIP

from RBInvParam.error_estimators.state_error_estimators import StateErrorEstimatorType
from RBInvParam.error_estimators.adjoint_error_estimators import AdjointErrorEstimatorType
from RBInvParam.error_estimators.objective_error_estimators import ObjectiveErrorEstimatorType

from RBInvParam.timestepping import TimeStepperType

from RBInvParam.utils.create_q_exact import *

from RBInvParam.optimizer import LoggerErrorChoice

#########################################################################################

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = Path('./dumps') / (timestamp + '_TR_IRGNM')
os.mkdir(save_path)
logfile_path= save_path / 'TR_IRGNM.log'

logger = get_default_logger(logger_name='TR_IRGNM',
                            logfile_path=logfile_path,
                            use_timestemp=False)
logger.setLevel(logging.DEBUG)

#########################################################################################''

set_log_levels({
    'pymor' : 'WARN'
})




# set_defaults({
#     'pymor.algorithms.gram_schmidt.gram_schmidt.atol': 1e-16,
#     'pymor.algorithms.gram_schmidt.gram_schmidt.rtol': 1e-16,
#     'pymor.algorithms.gram_schmidt.gram_schmidt.check_tol': 1e-6,
# })


#########################################################################################''
# from pymor.core.cache import clear_caches
# from pymor.core.cache import disable_caching

# disable_caching()
# clear_caches()
#########################################################################################''

def main():
    y_res = 30
    z_res = 30

    par_dim = (y_res + 1) * (z_res + 1)
    #* 5 * 3
    #par_dim = 3
    T_initial = 0
    #T_final = 10.0
    # T_final = 5.0
    # nt = 50

    T_final = 5.0
    nt = 50

    #T_final = 5.0
    #T_final = 10.0

    # T_final = 5.0
    # nt = 50


    # T_final = 10.0
    # nt = 100

    delta_t = (T_final - T_initial) / nt

    assert T_final > T_initial
    q_circ = np.ones((1, par_dim)) * 1
    q_exact = np.ones((1,par_dim)) * 1


    # q_exact[0,200] = 2
    # q_exact[0,300] = 3

    q_exact = q_exact[0,:].reshape(y_res+1,z_res+1)
    #q_exact[10:21,10:21] = 2
    # q_exact[9,21] = 3
    # q_exact[8,21] = 3
    # q_exact[7,21] = 3
    # q_exact[9,22] = 3
    # q_exact[8,22] = 3
    # q_exact[7,22] = 3
    # q_exact[9,20] = 3
    # q_exact[8,20] = 3
    # q_exact[7,20] = 3


    # q_exact[7,14] = 2
    # q_exact[6,14] = 2
    # q_exact[5,14] = 2
    # q_exact[7,13] = 2
    # q_exact[6,13] = 2
    # q_exact[5,13] = 2
    # q_exact[7,15] = 2
    # q_exact[6,15] = 2
    # q_exact[5,15] = 2

    # q_exact = q_exact.flatten()
    # q_exact = np.array([q_exact])


    # q_exact = q_exact[0,:].reshape(y_res+1,z_res+1)

    add_constant_patch(q_exact, center=(20, 15), value=3.0, half_size=0)
    add_constant_patch(q_exact, center=(6, 14), value=2.0, half_size=0)

    # #add_constant_patch(q_exact, center=(30, 20), value=3.0, half_size=1)
    # #add_constant_patch(q_exact, center=(10, 24), value=2.0, half_size=1)

    # #add_gaussian_patch(q_exact, center=(20, 15), sigma=2.0, amp=2.0, half_size=3)
    # #add_gaussian_patch(q_exact, center=(6, 14), sigma=2.0, amp=1.0, half_size=3)

    # import matplotlib.pyplot as plt
    # plt.imshow(q_exact)
    # plt.colorbar()
    # #plt.show()
    # plt.savefig('./q_exact.pdf')

    # import sys
    # sys.exit()
    
    q_exact = q_exact.flatten()
    q_exact = np.array([q_exact])

    
    #q_exact[0,100:300] = 3
    #q_exact[0,:] = 3
    #q_exact[0,:] = 3

    # q_exact = q_exact[0,:].reshape(y_res+1,z_res+1)
    # q_exact[0:15,0:15] = 3
    # q_exact = q_exact[0,:].reshape(y_res+1,z_res+1)
    # q_exact[0:20,0:20] = 3
    # q_exact = q_exact.flatten()
    # q_exact = np.array([q_exact])



    #q_exact[0,:] = 3

    # q_exact[0,450] = 2
    # q_exact[0,470] = 3


    #q_exact[0,50] = 2
    q_circ[0,:] = 1

    bounds = np.zeros((par_dim, 2))
    bounds[:,0] = 1e-20
    bounds[:,1] = 1e20

    setup = {
        'spatial_resolution' : [4,y_res,z_res],
        'body_force' : {
            'type' : mm.BodyForceType.CenterExcite,
            'hyperparameter' : {}
            # 'type' : mm.BodyForceType.Gaussian,
            # 'hyperparameter' : {
            #     'center': [-0.1,0.0,0.0],
            #     'sigma' : 2.0,
            # }
        },
        'system_matrix' : {
            'type' : mm.SystemMatrixType.CosseratDelamination,
            'hyperparameter' : {
                'lambda' : 1e1,
                'mu' : 1e1,
                # 'lambda' : 1e3,
                # 'mu' : 1e3,
                'nu' : 1e-3,
                #'nu' : 1e-1,
                'surface' : 'left'
            }
        },
        'observation_operator': {
            'type': mm.ObservationOperatorType.Identity,     # Type of observation operator (e.g., identity = full state observed)
            #'type': mm.ObservationOperatorType.Boundary,                       # Type of observation operator (e.g., identity = full state observed)
            #'type': mm.ObservationOperatorType.Sensors,                       # Type of observation operator (e.g., identity = full state observed)
            #'type': mm.ObservationOperatorType.SensorsGrid,                                   
            'hyperparameter' : {
                'spatial_resolution' : [4,y_res,z_res],
                # # #'radius' : 2.0,
                # 'radius' : 0.001,
                # 'second_row' : False 
                #'grid_sizes' : [2,8,8]
                #'grid_sizes' : [5,11,11]
            }
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
            #'prod_V': 'l2',
            'prod_C': 'euclid',                       # Product on C_h
        },
        'T_initial': T_initial,                       # Start time of the simulation
        'T_final': T_final,                           # End time of the simulation
        'delta_t': delta_t,                           # Time step size
        'noise_percentage': None,                     # Relative noise level, will be set by 'build_InstationaryModelIP'
        #'noise_level': 5 * 1e-3,                         # Absolute noise magnitude added to data
        'noise_level': 5 * 1e-5,                         # Absolute noise magnitude added to data
        'y_delta' : None,
        #'noise_level': 0.0,                      # Absolute noise magnitude added to data
        'q_circ': q_circ,                             # Backgroundlevel for the parameter
        'q_exact_function': None,                     # Exact parameter as function, will be set by 'build_InstationaryModelIP'
        'q_exact': q_exact,                           # Exact parameter values, will be set by 'build_InstationaryModelIP'
        'q_time_dep': False,                          # Whether parameter is time-dependent (bool)
        'riesz_rep_grad': True,                       # Use Riesz representative for gradient in optimization
        'riesz_rep_hess': False,                       # Use Riesz representative for gradient in optimization
        'bounds': bounds,                             # Bounds on parameter values (e.g., for optimization)
        'save_path' : save_path,
        'time_stepper' : {
            'primal' : {
                'type' : TimeStepperType.SecondOrderCrankNicolson,
                'config' : {
                    'zeta' : 0.5
                    #'zeta' : 1.0
                }
            },
            'adjoint' : {
                'type' : TimeStepperType.SecondOrderCrankNicolson,
                #'type' : TimeStepperType.SecondOrderCrankNicolsonAdjointDTO,
                'config' : {
                    'zeta' : 0.5
                    #'zeta' : 1.0
                }
            },
        }
    }

    FOM = build_InstationaryModelIP(setup, logger)
    q_exact = FOM.setup['q_exact']
    q_start = q_circ

    FOM.A.material_model.save_time_series(
        [v for v in FOM.A.material_model.force_list],
        str('rhs'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )

    u_exact = FOM.solve_state(FOM.Q.make_array(q_exact))

    FOM.A.material_model.save_time_series(
        [v.real_part.impl for v in u_exact.vectors],
        str('u_exact'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )

    p_exact = FOM.solve_adjoint(FOM.Q.make_array(q_exact), u = u_exact)
    FOM.A.material_model.save_time_series(
        [v.real_part.impl for v in p_exact.vectors],
        str('p_exact'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )

    u_start = FOM.solve_state(FOM.Q.make_array(q_start))
    FOM.A.material_model.save_time_series(
        [v.real_part.impl for v in u_start.vectors],
        str('u_start'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )

    p_start = FOM.solve_adjoint(FOM.Q.make_array(q_start), u = u_start)
    FOM.A.material_model.save_time_series(
        [v.real_part.impl for v in p_start.vectors],
        str('p_start'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )

    diff = u_start - u_exact    
    FOM.A.material_model.save_time_series(
        [v.real_part.impl for v in diff.vectors],
        str('diff'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )

    y_delta = FOM.C.range.from_numpy(setup['y_delta'])
    FOM.A.material_model.save_time_series(
        [v.real_part.impl for v in y_delta.vectors],
        str('y_delta'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )   

    diff_y_delta = y_delta - FOM.C.apply(u_start)
    FOM.A.material_model.save_time_series(
        [v.real_part.impl for v in diff_y_delta.vectors],
        str('diff_y_delta'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )

    _q_start = FOM.Q.make_array(q_start)
    J = FOM.compute_objective(_q_start)
    print(J)
    print(np.sqrt(2 * J))


    optimizer_parameter = {
        'q_0': q_start,                                              # Initial guess for the parameter to be optimized
        'alpha_0': 1e-5,                                              # Initial regularization parameter (data fidelity vs. regularization)
        #'alpha_0': 1e-10,                                              # Initial regularization parameter (data fidelity vs. regularization)
        'tol': 1e-9,                                                 # Absolute convergence tolerance for optimization
        'tau': 1.50,                                                  # Relative (to the noise) convergence tolerance for optimization
        #'tau': 3.50,                                                  # Relative (to the noise) convergence tolerance for optimization
        'noise_level': setup['noise_level'],                         # Noise level in observed data (from model setup)
        'theta': 0.40,
        'Theta': 1.95,                                               # Upper bound for step acceptance condition
        #'Theta': 1.50,                                               # Upper bound for step acceptance condition
        'tau_tilde': 3.5,                                            # Relative (to the noise) convergence tolerance for optimization inside the trust region
        #####################
        'i_max': 250,                                                 # Max number of outer optimization iterations
        'reg_loop_max': 10,                                          # Max number of regularization updates per iteration
        #'i_max_inner': 15,                                           # Max number of inner iterations
        'i_max_inner': 30,                                           # Max number of inner iterations
        'TR_armijo_max_iter': 5,                                     # Max iterations Armijo condition to enforce the trust-region
        'agc_armijo_max_iter': 50,                                  # Max iterations for computing the AGC
        #####################
        'use_error_estimator' : False,
        'use_adjoint_space' : False,
        #'use_adjoint_space' : True,
        'offline_parallel' : False,
        'reg_AGC_step' : False,
        #'TR_enforcement' : 'check_error',
        'TR_enforcement' : 'backtracking',
        #####################
        'lin_solver_parms': {
            'method': 'gd',                                          # Method for solving linear systems (e.g., gradient descent)
            'max_iter': 250,                                         # Maximum iterations for the linear solver
            #'lin_solver_tol': 5 * 1e-8,                                 # Convergence tolerance for the linear solver            
            #'lin_solver_tol': 5 * 1e-9,                                 # Convergence tolerance for the linear solver
            #'lin_solver_tol': 1e-12,                                 # Convergence tolerance for the linear solver
            'lin_solver_tol': 5 * 1e-9,                                 # Convergence tolerance for the linear solver
            'kappa_arm' : 1e-12,
            'armijo_inital_step_size': 1e-2,                                    # Initial step size for iterative linear solver
            'armijo_min_step_size' : 1e-20
        },
        'enrichment': {
            'parameter_basis' : {
                'reduced_basis' : True,
                'additional_snapshots' :{
                    'include_lin_grad' : False,
                    'include_each_nabla_J_time_step' : True,
                    'include_each_nabla_lin_J_time_step' : False,
                    'include_krylov_directions' : False,
                    # {
                    #     'n' : 5,
                    #     'inital_direction' : 'ones'
                    # },
                },
                'compression' : {
                    'normalize' : True,
                    'HaPOD' : {
                        'eps': 1e-1,
                        'omega' : 0.1,
                    },
                    # 'normalize' : None,
                    # 'HaPOD' : None,
                },
                #'coarsing' : None,
                'coarsing' : {
                    'rel_tol_coeff_nabla_J' : 1e-2,
                }
            },
            'state_basis' : {
                'additional_snapshots' :{
                    'include_lins' : False,
                    'include_krylov_sensitivites' : False,
                },
                'compression' : {
                    'normalize' : True,
                    'HaPOD' : {
                        'eps': 1e-3,
                        'omega' : 0.1,
                    },
                    # 'normalize' : None,
                    # 'HaPOD' : None,
                },
                #'coarsing' : None,
                'coarsing' : {
                    'rel_tol_coeff_u' : 1e-2,
                    'rel_tol_coeff_p' : 1e-2
                }
            },
            'adjoint_basis' : {
                'additional_snapshots' :{
                    'include_lins' : False,
                    'include_krylov_sensitivites' : False,
                },
                'compression' : {
                    'normalize' : True,
                    'HaPOD' : {
                        'HaPOD_tol': 1e-3,
                    },
                    # 'normalize' : None,
                    # 'HaPOD' : None,
                }
            }
        },
        'error_estimator_types' : {
            'state' : StateErrorEstimatorType.HYPERBOLIC,
            'adjoint' : AdjointErrorEstimatorType.NONE,
            'objective' : ObjectiveErrorEstimatorType.NAIVE,
        },
        'logging' : {
            'errors' : LoggerErrorChoice.OBJECTIVE,
        },
        #####################
        'use_cached_operators': False,                               # Reuse previously assembled operators to save computation
        'dump_every_nth_loop': 1,                                    # Dump intermediate results every n optimization iterations
        #####################
        # 'eta0': 0.05,                                                # Initial trust region tolerance
        # 'eta_min' : 1e-5,
        # 'eta_max' : 0.15,
        'eta0': 0.10,                                                # Initial trust region tolerance
        'eta_min' : 1e-5,
        'eta_max' : 0.30,
        'kappa_arm': 1e-12,                                          # Armijo condition constant for sufficient decrease
        'beta_1': 0.80,                                              # Trust region edge tolerance.
        'beta_2': 3/4,                                               # Tolerance for the trustworthiness.
        'beta_3': 0.5                                                # Shrinking/Enlarging factor for the trust region.
    }

    logger.info(f"Dumping model setup to {save_path / 'setup.pkl'}.")
    save_dict_to_pkl(path=save_path / 'setup.pkl',
                     data = setup,
                     use_timestamp=False)

    logger.info(f"Dumping model optimizer_parameter to {save_path / 'optimizer_parameter.pkl'}.")
    save_dict_to_pkl(path=save_path / 'optimizer_parameter.pkl',
                        data = optimizer_parameter,
                        use_timestamp=False)

    optimizer = QrVrROMOptimizer(
        FOM = FOM,
        optimizer_parameter = optimizer_parameter,
        logger = logger,
        save_path=save_path
    )
    q_est = optimizer.solve()
    # print(q_est)
    # u = FOM.solve_state(q_est)
    # FOM.A.material_model.save_time_series(
    #     [v.real_part.impl for v in u.vectors],
    #     str('u_est'),
    #     str(save_path),
    #     np.linspace(T_initial, T_final, nt+1)
    # )

    # diff = u - u_exact
    # FOM.A.material_model.save_time_series(
    #     [v.real_part.impl for v in diff.vectors],
    #     str('diff_est'),
    #     str(save_path),
    #     np.linspace(T_initial, T_final, nt+1)
    # )



if __name__ == '__main__':
    main()
