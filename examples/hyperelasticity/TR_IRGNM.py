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
from pymor.core.defaults import set_defaults, get_defaults
from pymor.algorithms.genericsolvers import solver_options

import RBInvParam.problems.hyperelasticity.hyperelasticity_model as hm
import RBInvParam.problems.shared.material_model as mm

#import material_model as mm
#import hyperelasticity_model as hm

from RBInvParam.optimizer.optimizer import QrVrROMOptimizer, LoggerErrorChoice
from RBInvParam.utils.io import save_dict_to_pkl
from RBInvParam.utils.logger import get_default_logger
from RBInvParam.problems.hyperelasticity.build import build_HyperElasticityModelIP

from RBInvParam.error_estimators.state_error_estimators import StateErrorEstimatorType
from RBInvParam.error_estimators.adjoint_error_estimators import AdjointErrorEstimatorType
from RBInvParam.error_estimators.objective_error_estimators import ObjectiveErrorEstimatorType
from RBInvParam.trust_region import TRType
from RBInvParam.schemas.reductor import LinearizationMethod

from RBInvParam.timestepping import TimeStepperType

from RBInvParam.utils.create_q_exact import *

#########################################################################################''

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
    'pymor.operators.constructions.LincombOperator' : 'ERROR',
    #'pymor.operators.constructions.AdjointOperator' : 'ERROR',
    #'pymor.algorithms.genericsolvers.lgmres' : 'ERROR',
    'pymor.algorithms' : 'ERROR'
})

set_defaults({
    #'pymor.algorithms.genericsolvers.solver_options.lgmres_tol' : 1e-12,
    #'pymor.algorithms.genericsolvers.solver_options.lgmres_maxiter' : int(1e3),
    #'pymor.algorithms.newton.newton.maxiter' : 1e3,
    #'pymor.algorithms.newton.newton.atol' : 1e-4,
    #'pymor.algorithms.newton.newton.atol' : 1e-5,
})



#########################################################################################''

# np.set_printoptions(linewidth=np.inf) 
# np.set_printoptions(threshold=np.inf)  # force full print

def main():
    p1 = (-0.1, -15.0, -15.0)
    p2 = ( 0.1,  15.0,  15.0)

    y_bounds = (p1[1], p2[1])
    z_bounds = (p1[2], p2[2])

    state_y_res = 10
    state_z_res = 10

    param_y_res = state_y_res
    param_z_res = state_z_res

    par_dim = (param_y_res + 1) * (param_z_res + 1) 
    T_initial = 0

    T_final = 5.0
    nt = 50 
    delta_t = (T_final - T_initial) / nt

    assert T_final > T_initial
    q_circ = np.ones((1, par_dim))
    q_exact = np.ones((1,par_dim))
    
    half_size = 0
    q_exact = q_exact[0,:].reshape(param_y_res+1,param_z_res+1)
    add_constant_patch_coords(q_exact, 
                              center_coords=( 5.0,  0.0), 
                              value=3.0, 
                              half_size=half_size,
                              y_bounds=y_bounds, 
                              z_bounds=z_bounds)

    add_constant_patch_coords(q_exact, 
                              center_coords=(-9.0, -1.0), 
                              value=2.0, 
                              half_size=half_size,
                              y_bounds=y_bounds, 
                              z_bounds=z_bounds)

    q_exact = q_exact.flatten()
    q_exact = np.array([q_exact])

    q_circ[0,:] = 1.0

    bounds = np.zeros((par_dim, 2))
    bounds[:,0] = 1e-20
    bounds[:,1] = 1e20

    state_grid_resolution = [4,state_y_res,state_z_res]
    param_grid_resolution = [4,param_y_res,param_z_res]

    setup = {
        'p1' : p1,
        'p2' : p2,
        'param_grid_resolution' : param_grid_resolution,
        'state_grid_resolution' : state_grid_resolution,
        'body_force' : {
            'type' : mm.BodyForceType.CenterExcite,
            'hyperparameter' : {}
        },
        'stored_energy' : {
            'type' : hm.StoredEnergyFunctionType.Hookean,
            #'type' : hm.StoredEnergyFunctionType.NeoHookean,
            'hyperparameter' : {
                # 'mu' : 26.32, 
                # 'kappa' : 68.60
                'mu' : 1e1, 
                'lambda' : 1e1
            }
        },
        'boundary_condition' : {
            'type': mm.BoundaryConditionType.DirichletOnYandZ,
            'hyperparameter' : {}
        },
        'observation_operator': {
            #'type': mm.ObservationOperatorType.Identity,                       # Type of observation operator (e.g., identity = full state observed)
            'type': mm.ObservationOperatorType.Sensors,
            'hyperparameter' : {
                'spatial_resolution' : state_grid_resolution,
                'radius' : 0.001,
                'second_row' : False 
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
            #'prod_Q': 'euclid',                      # Product on Q_h
            'prod_Q': 'h1',                           # Product on Q_h
            'prod_V': 'h1_0_semi',                    # Product on V_h
            'prod_C': 'euclid',                       # Product on C_h
        },
        'T_initial': T_initial,                       # Start time of the simulation
        'T_final': T_final,                           # End time of the simulation
        'delta_t': delta_t,                           # Time step size
        'noise_percentage': None,                     # Relative noise level, will be set by 'build_InstationaryModelIP'
        'noise_level': 5 * 1e-5,                      # Absolute noise magnitude added to data
        #'noise_level': 0,                      # Absolute noise magnitude added to data
        'q_circ': q_circ,                             # Backgroundlevel for the parameter
        'q_exact_function': None,                     # Exact parameter as function, will be set by 'build_InstationaryModelIP'
        'q_exact': q_exact,                           # Exact parameter values, will be set by 'build_InstationaryModelIP'
        'q_time_dep': False,                          # Whether parameter is time-dependent (bool)
        'riesz_rep_grad': True,                       # Use Riesz representative for gradient in optimization
        'riesz_rep_hess': False,                       
        'bounds': bounds,                             # Bounds on parameter values (e.g., for optimization)
        'save_path' : save_path,
        'time_stepper' : {
            'state' : {
                #'type' : TimeStepperType.SecondOrderCrankNicolson,
                'type' : TimeStepperType.SecondOrderCrankNicolson,
                'config' : {
                    'zeta' : 0.5
                }
            },
            'adjoint' : {
                'type' : TimeStepperType.SecondOrderCrankNicolson,
                #'type' : TimeStepperType.SecondOrderCrankNicolsonAdjointDTO,
                'config' : {
                    'zeta' : 0.5
                }
            },
            'lin_state' : {
                'type' : TimeStepperType.SecondOrderCrankNicolson,
                'config' : {
                    'zeta' : 0.5
                }
            },
            'lin_adjoint' : {
                'type' : TimeStepperType.SecondOrderCrankNicolson,
                #'type' : TimeStepperType.SecondOrderCrankNicolsonAdjointDTO,
                'config' : {
                    'zeta' : 0.5
                }
            },
        }
    }


    FOM = build_HyperElasticityModelIP(setup, logger)
    q_exact = FOM.setup['q_exact']
    q_start = q_circ

    # _q_start = FOM.Q.make_array(q_start)
    # print(FOM.compute_objective(_q_start))

    # for i in range(8):
    #     q_exact = np.ones((1,par_dim))
    #     q_exact = q_exact[0,:].reshape(param_y_res+1,param_z_res+1)
    #     q_exact[:, i:8] = 2.0
    #     q_exact = q_exact.flatten()
    #     q_exact = np.array([q_exact])    

    # #print(q_exact)

    #     _q_exact = FOM.Q.make_array(q_exact)
    #     print(FOM.compute_objective(_q_exact))
    # import sys
    # sys.exit()

    # q_exact = np.ones((1,par_dim))
    # _q_exact = FOM.Q.make_array(q_exact)
    # print(FOM.compute_objective(_q_exact))

    # q_exact = 2 * np.ones((1,par_dim))
    # _q_exact = FOM.Q.make_array(q_exact)
    # print(FOM.compute_objective(_q_exact))

    # import sys
    # sys.exit()

    u_exact = FOM.solve_state(FOM.Q.make_array(q_exact))
    FOM.A.hyperelasticity_model.save_time_series(
        [v.impl for v in u_exact.vectors],
        str('u_exact'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )

    p_exact = FOM.solve_adjoint(FOM.Q.make_array(q_exact), u = u_exact)
    FOM.A.hyperelasticity_model.save_time_series(
        [v.impl for v in p_exact.vectors],
        str('p_exact'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )

    u_start = FOM.solve_state(FOM.Q.make_array(q_start))
    FOM.A.hyperelasticity_model.save_time_series(
        [v.impl for v in u_start.vectors],
        str('u_start'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )


    p_start = FOM.solve_adjoint(FOM.Q.make_array(q_start), u = u_start)
    FOM.A.hyperelasticity_model.save_time_series(
        [v.impl for v in p_start.vectors],
        str('p_start'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )

    diff = u_start - u_exact
    FOM.A.hyperelasticity_model.save_time_series(
        [v.impl for v in diff.vectors],
        str('diff'),
        str(save_path),
        np.linspace(T_initial, T_final, nt+1)
    )

    # _q_start = FOM.Q.make_array(q_start)
    # _q_exact = FOM.Q.make_array(q_exact)
    # J = FOM.compute_objective(_q_start)
    
    # print(FOM.compute_objective(_q_start))
    # print(FOM.compute_objective(_q_exact))
    # _d = FOM.Q.zeros()
    # print(FOM.compute_linearized_objective(_q_start, _d, 0.0))
    # print(FOM.compute_linearized_objective(_q_exact, _d, 0.0))
    # print(np.sqrt(2 * J))

    # print(FOM.compute_gradient(_q_start))
    # print(FOM.compute_gradient(_q_exact))


    optimizer_parameter = {
        'method' : 'TR_IRGNM',
        'q_0': q_start,                                              # Initial guess for the parameter to be optimized
        'alpha_0': 1e-5,                                              # Initial regularization parameter (data fidelity vs. regularization)
        #'alpha_0': 1e-10,                                              # Initial regularization parameter (data fidelity vs. regularization)
        'tol': 1e-9,                                                 # Absolute convergence tolerance for optimization
        'tau': 1.50,                                                  # Relative (to the noise) convergence tolerance for optimization
        #'tau': 1.00,                                                  # Relative (to the noise) convergence tolerance for optimization
        'noise_level': setup['noise_level'],                         # Noise level in observed data (from model setup)
        'theta': 0.40,
        'Theta': 1.95,                                               # Upper bound for step acceptance condition
        #'Theta': 1.50,                                               # Upper bound for step acceptance condition
        'tau_tilde': 3.5,                                            # Relative (to the noise) convergence tolerance for optimization inside the trust region
        #####################
        'i_max': 250,                                                 # Max number of outer optimization iterations
        'reg_loop_max': 10,                                          # Max number of regularization updates per iteration
        'reg_loop_max': 30,                                          # Max number of regularization updates per iteration
        #'i_max_inner': 15,                                           # Max number of inner iterations
        'i_max_inner': 30,                                           # Max number of inner iterations
        'AGC_armijo_cfg' : {
            "max_iter": 50,
            "initial_step_size": 1.0,
            "kappa_arm": 1e-12,
            "shrink": 0.5,
        },
        'TR_armijo_cfg' : {
            "max_iter": 5,
            "initial_step_size": 1.0,
            "kappa_arm": 1e-12,
            "shrink": 0.5,
        },
        'TR': {
            'type': TRType.RELATIVE_OBJECTIVE_ERROR,

            # TR config
            'eta_initial': 0.15,        
            'eta_min': 1e-5,
            'eta_max': 0.30,
            'beta_1': 0.80,
            'beta_2': 0.80,
            'beta_3': 0.75,
        },
        # 'TR': {
        #     'type': TRType.RADIUS,

        #     # TR config
        #     'eta_initial': 0.25,        
        #     'eta_min': 1e-2,
        #     'eta_max': 1.00,
        #     'beta_1': 0.80,
        #     'beta_2': 0.80,
        #     'beta_3': 0.75,
        # },
        #####################
        'use_cached_operators': True,                               # Reuse previously assembled operators to save computation
        'use_error_estimator' : False,
        'reg_AGC_step' : False,
        'TR_enforcement' : 'backtracking',
        'dump_every_nth_loop': 1,                                    # Dump intermediate results every n optimization iterations
        'reductor' : {
            #'type' : 'default',
            'type' : 'material_model',
            'use_adjoint_space' : False,
            'offline_parallel' : False,
            'error_estimator_types' : {
                'state' : StateErrorEstimatorType.HYPERBOLIC,
                'adjoint' : AdjointErrorEstimatorType.NONE,
                'objective' : ObjectiveErrorEstimatorType.NAIVE,
            },
            'check_orthonormality' : True,
            'check_tol' : 1e-9,
            'linearization_method' : LinearizationMethod.DEIM,
        },
        #####################
        'lin_solver_parms': {
            'method': 'gd',                                          # Method for solving linear systems (e.g., gradient descent)
            'max_iter': 250,                                         # Maximum iterations for the linear solver
            #'abs_grad_tol' : 5 * 1e-9,
            'abs_grad_tol' : 5 * 1e-11,
            'rel_change_obj_tol' : 1e-4,
            'kappa_arm' : 1e-12,
            'armijo_inital_step_size': 1e-2,                                    # Initial step size for iterative linear solver
            'armijo_min_step_size' : 1e-20
        },
        'enrichment': {
            'parameter_basis' : {
                'additional_snapshots' :{
                    'include_lin_grad' : False,
                    'include_each_nabla_J_time_step' : True,
                    'include_each_nabla_lin_J_time_step' : False,
                    'include_krylov_directions' : False,
                },
                'compression' : {
                    'normalize' : True,
                    'HaPOD' : 
                    {
                        'eps': 1e-1,
                        'omega' : 0.1,
                    },
                },
                'coarsing' : None,
            },
            'state_basis' : {
                'additional_snapshots' :{
                    'include_lin_states' : False,
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
                'coarsing' : None,
                # 'coarsing' : {
                #     'rel_tol_coeff_u' : 1e-2,
                #     'rel_tol_coeff_p' : 1e-2
                # }
            },
            #'state_basis' : None,
            'adjoint_basis' : None
            
        },
        'logging' : {
            'errors' : LoggerErrorChoice.OBJECTIVE,
        },        
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
   


if __name__ == '__main__':
    main()
