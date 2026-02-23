import numpy as np
import copy

import RBInvParam.problems.hyperelasticity.hyperelasticity_model as hm
import RBInvParam.problems.shared.material_model as mm

from RBInvParam.error_estimators.state_error_estimators import StateErrorEstimatorType
from RBInvParam.error_estimators.adjoint_error_estimators import AdjointErrorEstimatorType
from RBInvParam.error_estimators.objective_error_estimators import ObjectiveErrorEstimatorType
from RBInvParam.trust_region import TRType
from RBInvParam.schemas.reductor import LinearizationMethod

from RBInvParam.timestepping import TimeStepperType

from RBInvParam.utils.create_q_exact import *

from RBInvParam.optimizer.optimizer import LoggerErrorChoice

p1 = (-0.1, -15.0, -15.0)
p2 = ( 0.1,  15.0,  15.0)

y_bounds = (p1[1], p2[1])
z_bounds = (p1[2], p2[2])


state_y_res = 30
state_z_res = 30

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
        'prod_Q': 'euclid',                       # Product on Q_h
        'prod_V': 'h1_0_semi',                    # Product on V_h
        'prod_C': 'euclid',                       # Product on C_h
    },
    'T_initial': T_initial,                       # Start time of the simulation
    'T_final': T_final,                           # End time of the simulation
    'delta_t': delta_t,                           # Time step size
    'noise_percentage': None,                     # Relative noise level, will be set by 'build_InstationaryModelIP'
    'noise_level': 5  * 1e-5,                      # Absolute noise magnitude added to data
    'q_circ': q_circ,                             # Backgroundlevel for the parameter
    'q_exact_function': None,                     # Exact parameter as function, will be set by 'build_InstationaryModelIP'
    'q_exact': q_exact,                           # Exact parameter values, will be set by 'build_InstationaryModelIP'
    'q_time_dep': False,                          # Whether parameter is time-dependent (bool)
    'riesz_rep_grad': True,                       # Use Riesz representative for gradient in optimization
    'riesz_rep_hess': False,                       # Use Riesz representative for gradient in optimization
    'bounds': bounds,                             # Bounds on parameter values (e.g., for optimization)
    'save_path' : None,
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

q_start = q_circ
lin_solver_tol = 5 * 1e-9
tau = 1.50

FOM_optimizer_parameter = {
    'method' : 'FOM_IRGNM',
    'q_0': q_start,                                          # Initial guess for the parameter to be optimized
    'alpha_0': 1e-5,                                          # Initial regularization parameter
    'tol': 1e-9,                                            # Absolute convergence tolerance for optimization
    'tau': tau,                                              # Relative (to the noise) convergence tolerance for optimization
    'noise_level': setup['noise_level'],                     # Noise level in observed data (from model setup)
    'theta': 0.4,                                         # Lower tolerance for the direction acceptance condition
    'Theta': 1.95,                                           # Upper tolerance for the direction acceptance condition
    #####################
    'i_max': 250,                                             # Maximum number of outer optimization iterations
    'reg_loop_max': 25,                                      # Maximum number of regularization updates per step
    'i_max_inner': 10,                                       # Maximum number of inner iterations
    ####################
    'lin_solver_parms': {
        'method': 'gd',                                          # Method for solving linear systems (e.g., gradient descent)
        'max_iter': 250,                                         # Maximum iterations for the linear solver
        'abs_grad_tol' : 5 * 1e-9,        
        'rel_change_obj_tol' : 1e-4,
        'kappa_arm' : 1e-12,
        'armijo_inital_step_size': 1e-2,                                    # Initial step size for iterative linear solver
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
    'tau': tau,                                                  # Relative (to the noise) convergence tolerance for optimization
    'noise_level': setup['noise_level'],                         # Noise level in observed data (from model setup)
    'theta': 0.4,
    'Theta': 1.95,                                               # Upper bound for step acceptance condition
    #'Theta': 1.50,                                               # Upper bound for step acceptance condition
    'tau_tilde': 3.5,                                            # Relative (to the noise) convergence tolerance for optimization inside the trust region
    #####################
    'i_max': 250,                                                 # Max number of outer optimization iterations
    'reg_loop_max': 10,                                          # Max number of regularization updates per iteration
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
    },                                 # Max iterations Armijo condition to enforce the trust-region 
    #####################
    'use_cached_operators': True,                               # Reuse previously assembled operators to save computation
    'use_error_estimator' : False,
    'reg_AGC_step' : False,
    'TR_enforcement' : 'backtracking',
    'dump_every_nth_loop': 1,                                    # Dump intermediate results every n optimization iterations
    'reductor' : {
        'type' : 'default',
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
        'abs_grad_tol' : 5 * 1e-9,
        'rel_change_obj_tol' : 1e-4,
        'kappa_arm' : 1e-12,
        'armijo_inital_step_size': 1e-2,                                    # Initial step size for iterative linear solver
        'armijo_min_step_size' : 1e-20
    },
    'enrichment': {
        'parameter_basis' : {
            'reduced_basis' : True,
            'additional_snapshots' : {
                'include_lin_grad' : False,
                'include_each_nabla_J_time_step' : False,
                'include_each_nabla_lin_J_time_step' : False,
                'include_krylov_directions' : False,
            },
            'compression' : {
                'normalize' : None,
                'HaPOD' : None,
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
            },
            'coarsing' : None,
        },
        'adjoint_basis' : None
    },
    'logging' : {
        'errors' : LoggerErrorChoice.OBJECTIVE,
    }                                         # Shrinking/Enlarging factor for the trust region.
}

EXPERIMENTS = {}

# setup_identity['noise_level'] = 2.5 * 1e-4

# identity_lin_solver_tol = 5 * 1e-9
# grid_lin_solver_tol = 5 * 1e-9

identity_lin_solver_tol = lin_solver_tol
grid_lin_solver_tol = lin_solver_tol


#----------------------------------------------------------------------------------------

FOM_optimizer_parameter_ = copy.deepcopy(FOM_optimizer_parameter)
TR_optimizer_parameter_ = copy.deepcopy(TR_optimizer_parameter)

##########################################################################################

setup_sensors = copy.deepcopy(setup)
setup_grid = copy.deepcopy(setup)
setup_identity = copy.deepcopy(setup)

setup_identity['observation_operator']['type'] = mm.ObservationOperatorType.Identity
setup_identity['observation_operator']['hyperparameter'] = {}

setup_grid['observation_operator']['type'] = mm.ObservationOperatorType.SensorsGrid
setup_grid['observation_operator']['hyperparameter'] = {
    'radius' : 0.001,
    'grid_sizes' : [2,8,8]
}

#----------------------------------------------------------------------------------------

FOM_optimizer_parameter_sensors = copy.deepcopy(FOM_optimizer_parameter_)
FOM_optimizer_parameter_grid = copy.deepcopy(FOM_optimizer_parameter_)
FOM_optimizer_parameter_identity = copy.deepcopy(FOM_optimizer_parameter_)

#FOM_optimizer_parameter_identity['noise_level'] = setup_identity['noise_level']
FOM_optimizer_parameter_identity['lin_solver_parms']['lin_solver_tol'] = identity_lin_solver_tol
FOM_optimizer_parameter_grid['lin_solver_parms']['lin_solver_tol'] = grid_lin_solver_tol

EXPERIMENTS['FOM_sensors'] = (setup_sensors, FOM_optimizer_parameter_sensors)
EXPERIMENTS['FOM_identity'] = (setup_identity, FOM_optimizer_parameter_identity)
EXPERIMENTS['FOM_grid'] = (setup_grid, FOM_optimizer_parameter_grid)

#----------------------------------------------------------------------------------------

TR_optimizer_parameter__ = copy.deepcopy(TR_optimizer_parameter_)
TR_optimizer_parameter__['enrichment']['parameter_basis']['additional_snapshots']['include_each_nabla_J_time_step'] = True
TR_optimizer_parameter__['enrichment']['parameter_basis']['compression']['normalize'] = True
TR_optimizer_parameter__['enrichment']['parameter_basis']['compression']['HaPOD'] = {'eps': 1e-1, 'omega' : 0.1}

TR_optimizer_parameter_sensors = copy.deepcopy(TR_optimizer_parameter__)
TR_optimizer_parameter_grid = copy.deepcopy(TR_optimizer_parameter__)
TR_optimizer_parameter_identity = copy.deepcopy(TR_optimizer_parameter__)

#TR_optimizer_parameter_identity['noise_level'] = setup_identity['noise_level']
TR_optimizer_parameter_identity['lin_solver_parms']['lin_solver_tol'] = identity_lin_solver_tol
TR_optimizer_parameter_grid['lin_solver_parms']['lin_solver_tol'] = grid_lin_solver_tol

EXPERIMENTS['TR_sensors_time_step'] = (setup_sensors, TR_optimizer_parameter_sensors)
EXPERIMENTS['TR_identity_time_step'] = (setup_identity, TR_optimizer_parameter_identity)
EXPERIMENTS['TR_grid_time_step'] = (setup_grid, TR_optimizer_parameter_grid)


#----------------------------------------------------------------------------------------

TR_optimizer_parameter__ = copy.deepcopy(TR_optimizer_parameter_)
TR_optimizer_parameter__['enrichment']['parameter_basis']['additional_snapshots']['include_each_nabla_J_time_step'] = True

TR_optimizer_parameter_sensors = copy.deepcopy(TR_optimizer_parameter__)
TR_optimizer_parameter_grid = copy.deepcopy(TR_optimizer_parameter__)
TR_optimizer_parameter_identity = copy.deepcopy(TR_optimizer_parameter__)

#TR_optimizer_parameter_identity['noise_level'] = setup_identity['noise_level']
TR_optimizer_parameter_identity['lin_solver_parms']['lin_solver_tol'] = identity_lin_solver_tol
TR_optimizer_parameter_grid['lin_solver_parms']['lin_solver_tol'] = grid_lin_solver_tol

EXPERIMENTS['TR_sensors_time_step_full'] = (setup_sensors, TR_optimizer_parameter_sensors)
EXPERIMENTS['TR_identity_time_step_full'] = (setup_identity, TR_optimizer_parameter_identity)
EXPERIMENTS['TR_grid_time_step_full'] = (setup_grid, TR_optimizer_parameter_grid)


#----------------------------------------------------------------------------------------

TR_optimizer_parameter__ = copy.deepcopy(TR_optimizer_parameter_)

TR_optimizer_parameter_sensors = copy.deepcopy(TR_optimizer_parameter__)
TR_optimizer_parameter_grid = copy.deepcopy(TR_optimizer_parameter__)
TR_optimizer_parameter_identity = copy.deepcopy(TR_optimizer_parameter__)

#TR_optimizer_parameter_identity['noise_level'] = setup_identity['noise_level']
TR_optimizer_parameter_identity['lin_solver_parms']['lin_solver_tol'] = identity_lin_solver_tol
TR_optimizer_parameter_grid['lin_solver_parms']['lin_solver_tol'] = grid_lin_solver_tol

EXPERIMENTS['TR_sensors'] = (setup_sensors, TR_optimizer_parameter_sensors)
EXPERIMENTS['TR_identity'] = (setup_identity, TR_optimizer_parameter_identity)
EXPERIMENTS['TR_grid'] = (setup_grid, TR_optimizer_parameter_grid)


#----------------------------------------------------------------------------------------

# TR_optimizer_parameter__ = copy.deepcopy(TR_optimizer_parameter_)
# TR_optimizer_parameter__['enrichment']['parameter_basis']['additional_snapshots']['include_each_nabla_lin_J_time_step'] = True
# TR_optimizer_parameter__['enrichment']['parameter_basis']['compression']['normalize'] = True
# TR_optimizer_parameter__['enrichment']['parameter_basis']['compression']['HaPOD'] = {'eps': 1e-1, 'omega' : 0.1}

# TR_optimizer_parameter_sensors = copy.deepcopy(TR_optimizer_parameter__)
# TR_optimizer_parameter_grid = copy.deepcopy(TR_optimizer_parameter__)
# TR_optimizer_parameter_identity = copy.deepcopy(TR_optimizer_parameter__)

# #TR_optimizer_parameter_identity['noise_level'] = setup_identity['noise_level']
# TR_optimizer_parameter_identity['lin_solver_parms']['lin_solver_tol'] = identity_lin_solver_tol
# TR_optimizer_parameter_grid['lin_solver_parms']['lin_solver_tol'] = grid_lin_solver_tol

# EXPERIMENTS['TR_sensors_time_step_lin'] = (setup_sensors, TR_optimizer_parameter_sensors)
# EXPERIMENTS['TR_identity_time_step_lin'] = (setup_identity, TR_optimizer_parameter_identity)
# EXPERIMENTS['TR_grid_time_step_lin'] = (setup_grid, TR_optimizer_parameter_grid)

prefix = 'elasticity'
EXPERIMENTS = {f"{prefix}_{k}": v for k, v in EXPERIMENTS.items()}