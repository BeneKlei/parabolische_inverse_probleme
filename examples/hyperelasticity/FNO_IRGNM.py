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

from RBInvParam.models.FNO.FNO import FNO1d_new
from RBInvParam.models.FNO.utils import generate_training_data
from RBInvParam.models.FNO.training import training
from torch.utils.data import DataLoader, random_split

import torch

#########################################################################################''

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = Path('./dumps') / (timestamp + '_FNO_IRGNM')
os.mkdir(save_path)
logfile_path= save_path / 'FNO_IRGNM.log'

logger = get_default_logger(logger_name='FNO_IRGNM',
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

    center = (
        p1[0],
        p1[1] + (p2[1] - p1[1]) / 2,
        p1[2] + (p2[2] - p1[2]) / 2
    )

    y_bounds = (p1[1], p2[1])
    z_bounds = (p1[2], p2[2])

    state_x_res = 4
    state_y_res = 20
    state_z_res = 20

    # state_y_res = 60
    # state_z_res = 60

    param_x_res = state_x_res
    param_y_res = state_y_res
    param_z_res = state_z_res

    par_dim = (param_y_res + 1) * (param_z_res + 1)

    #################################################
    # Set:
    # 1 PU = 10^9 GPa
    # 1 LU = 1/30m
    # 1 DU = 10^3 kg m^{-3}
    # Derived
    # 1 TU = 3.33 * 10^-5s
    #################################################

    T_initial = 0
    T_final = 16.0
    nt = 64
    

    delta_t = (T_final - T_initial) / nt

    rho_hat = 2.70
    #rho_hat = 1.00

    assert T_final > T_initial
    q_circ = np.ones((1, par_dim))
    q_exact = np.ones((1,par_dim))

    #
    q_exact = q_exact[0,:].reshape(param_y_res+1,param_z_res+1)

    # --------------------------------------------------------------------------
    # half_size = 1
    # add_constant_square_patch_from_center_coords(q_exact, 
    #                           center_coords=( 5.0,  0.0), 
    #                           value=3.0, 
    #                           half_size=half_size,
    #                           y_bounds=y_bounds, 
    #                           z_bounds=z_bounds)


    # add_constant_square_patch_from_center_coords(q_exact, 
    #                           center_coords=(-9.0, -1.0), 
    #                           value=2.0, 
    #                           half_size=half_size,
    #                           y_bounds=y_bounds, 
    #                           z_bounds=z_bounds)

    # --------------------------------------------------------------------------
    half_size = 1
    add_constant_square_patch_from_center_coords(q_exact, 
                            center_coords=( 5.0,  0.0), 
                            value=3.0, 
                            half_size=half_size,
                            interpolated=True,
                            distance="square",
                            y_bounds=y_bounds, 
                            z_bounds=z_bounds)


    add_constant_square_patch_from_center_coords(q_exact, 
                            center_coords=(-9.0, -1.0), 
                            value=2.0, 
                            half_size=half_size,
                            interpolated=True,
                            distance="square",
                            y_bounds=y_bounds, 
                            z_bounds=z_bounds)

    # --------------------------------------------------------------------------
    # half_size = 1
    # add_constant_square_patch_from_center_coords(q_exact, 
    #                         center_coords=( 1.0,  -1.0), 
    #                         value=3.0, 
    #                         half_size=half_size,
    #                         y_bounds=y_bounds, 
    #                         z_bounds=z_bounds)

    # --------------------------------------------------------------------------
    # half_size = 1
    # add_constant_square_patch_from_center_coords(q_exact, 
    #                           center_coords=( 1.0,  -10.0), 
    #                           value=3.0, 
    #                           half_size=half_size,
    #                           y_bounds=y_bounds, 
    #                           z_bounds=z_bounds)


    # add_constant_square_patch_from_center_coords(q_exact, 
    #                           center_coords=( -5.0,  7.0), 
    #                           value=3.0, 
    #                           half_size=half_size,
    #                           y_bounds=y_bounds, 
    #                           z_bounds=z_bounds)


    # add_constant_square_patch_from_center_coords(q_exact, 
    #                           center_coords=( 8.0,  9.0), 
    #                           value=3.0, 
    #                           half_size=half_size,
    #                           y_bounds=y_bounds, 
    #                           z_bounds=z_bounds)


    # add_constant_rect_patch_from_corners_coords(
    #     q_exact,
    #     tl_coords = (-10, -10),
    #     br_coords = (-10 + 5, -10 + 20),
    #     value = 0.5,
    #     y_bounds=y_bounds,
    #     z_bounds=z_bounds
    # )

    # import matplotlib.pyplot as plt
    # plt.imshow(q_exact)
    # plt.colorbar()
    # plt.savefig('./q_exact.pdf')


    q_exact = q_exact.flatten()
    q_exact = np.array([q_exact])


    parameter_factor = 1
    q_exact = parameter_factor * q_exact.flatten()
    q_exact = np.array([q_exact])

    q_circ[0,:] = parameter_factor * 1.0

    bounds = np.zeros((par_dim, 2))
    bounds[:,0] = 1e-20
    bounds[:,1] = 1e20

    state_grid_resolution = [state_x_res,state_y_res,state_z_res]
    param_grid_resolution = [param_x_res,param_y_res,param_z_res]

    setup = {
        'p1' : p1,
        'p2' : p2,
        'param_grid_resolution' : param_grid_resolution,
        'state_grid_resolution' : state_grid_resolution,
        'body_force' : {
            'type' : mm.BodyForceType.SharpPulse,
            'hyperparameter' : {
                'origin' : center,
                'end_time' : 0.5,
                'factor' : (1.0 / rho_hat),
                'width' : 1.00
            }
            # 'type' : mm.BodyForceType.WavePulse,
            # 'hyperparameter' : {
            #     'origin' : center,
            #     'end_time' : 4 * 1e-5, # physical time
            #     'factor' : 1 / rho_hat,
            #     'time_scaling_factor' : 3.33 * 10^-5
            # }
        },
        'stored_energy' : {
            'type' : hm.StoredEnergyFunctionType.Hookean,
            #'type' : hm.StoredEnergyFunctionType.NeoHookean,
            'hyperparameter' : {
                # 'mu' : 26.32,
                # 'kappa' : 68.60
                # 'mu' : (5.6 / rho_hat),
                # 'lambda' : (10.9 / rho_hat),
                # 'mu' : (5.6 / rho_hat),
                # 'lambda' : (10.9 / rho_hat),
                'mu' : 1/parameter_factor * (11.2 / rho_hat),
                'lambda' : 1/parameter_factor * (21.8 / rho_hat),
                # 'mu' : 4 * 4.15,
                # 'lambda' : 4 * 8.07
            }
        },
        'boundary_condition' : {
            'type': mm.BoundaryConditionType.DirichletOnYandZ,
            'hyperparameter' : {}
        },
        'observation_operator': {
            'type': mm.ObservationOperatorType.Identity,                       # Type of observation operator (e.g., identity = full state observed)
            'hyperparameter' : {},
            # 'type': mm.ObservationOperatorType.Sensors,
            # 'hyperparameter' : {
            #     'p1' : p1,
            #     'p2' : p2,
            #     'sensor_patch_size' : (28.0, 28.0),
            #     'sensor_spacing' : 1.0,
            #     'at_top' : True,
            #     'at_bottom' : False,
            #     'sensor_patch_center_offset' : (0.0, 0.0),
            #     'x_face_offset' : 0.00,
            #     'radius' : 0.001,
            #     'use_boundary_mass_matrix' : True,
            # }
        },
        'dims' : {
            'nt': nt,                                     # Number of time steps
            'par_dim' : None,
            'state_dim' : None,
            'observation_space_dim': None
        },
        'products': {                                 # Inner products used in the problem
            'prod_H': 'l2',                           # Product on H_h
            'prod_Q': 'euclid',                      # Product on Q_h
            'prod_V': 'h1',                           # Product on V_h
            'prod_C': 'euclid',                       # Product on C_h
            'prod_reg' : 'euclid'
        },
        'T_initial': T_initial,                       # Start time of the simulation
        'T_final': T_final,                           # End time of the simulation
        'delta_t': delta_t,                           # Time step size
        'noise_percentage': None,                     # Relative noise level, will be set by 'build_InstationaryModelIP'
        #'noise_level': 5 * 1e-4,                      # Absolute noise magnitude added to data
        #'noise_level': 5 * 1e-5,                      # Absolute noise magnitude added to data
        #'noise_level': 0,                      # Absolute noise magnitude added to data
        'noise_info' : {
            #'noise_level_input' : 1.0 * 1e-2,
            'noise_level_input' : 0.0,
            'noise_level_mode' : 'rel',
            'abs_noise_level_y' : None,
            'rel_noise_level_y' : None,
            'y_norm' : None,
        },
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

    n_samples = 10

    X = np.zeros((n_samples, nt + 1, par_dim), dtype=np.float32)
    for i in range(n_samples):
        X[i] = q_exact.flatten()


    dataset, X, Y, data_path = generate_training_data(
        FOM=FOM,
        n_samples=n_samples,
        param_y_res=param_y_res,
        param_z_res=param_z_res,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        nt=nt,
        save_path=save_path,
        seed=42,
        X = X
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    torch.save(train_dataset, data_path / "train_dataset.pt")
    torch.save(val_dataset, data_path / "val_dataset.pt")


    fno = FNO1d_new(
        dim_Q = FOM.Q.dim,
        dim_V = FOM.V.dim,
        modes = 16,
        lifting_width = FOM.Q.dim,
    )

    training(
        train_loader = train_loader,
        val_loader = val_loader,
        model = fno,
        checkpoint_path = data_path,
        num_epoch = 50
    )  

    # --------------------------------------------
    
    u_exact = FOM.solve_state(FOM.Q.make_array(q_exact))
    u_NN = fno(
        torch.tensor(np.expand_dims(
            np.tile(q_exact, (nt + 1, 1)), 
            axis = 0
        ),    
        dtype=torch.float32
    ))
    

    # u_NN = torch.tensor(u_NN, dtype=torch.float64)
    # u_NN_ = FOM.V.empty()

    # import pymor_dealii_bindings as pd2
    # for v in u_NN[0].detach().numpy():
    #     u_NN_.append(pd2.Vector(v))

    u_exact_np = u_exact.to_numpy()
    u_NN_np = u_NN.detach().cpu().numpy().astype(np.float64)

    #print(u_exact_np - u_NN_np)
    diff = (u_exact_np - u_NN_np)[0]
    norm_diff = np.linalg.norm(diff, axis=1)
    print(norm_diff)
    print(norm_diff / np.linalg.norm(u_exact_np, axis=1))

    


    # u_NN_ = FOM.V.empty()

    # for i, values in enumerate(u_NN_np[0]):
    #     vec = u_exact.vectors[i].copy()
    #     vec.scal(0.0)
    #     vec.axpy(1.0, FOM.V.make_array(values.reshape(1, -1))[0])
    #     u_NN_.append(vec)
    
    # #FOM.V.make_array(u_NN[0].detach().numpy())

    # print(u_exact.to_numpy()[0].dtype)
    # print(u_NN_.to_numpy()[0].dtype)

    
    # --------------------------------------------

    # FOM.A.hyperelasticity_model.save_time_series(
    #     [v.impl for v in u_exact.vectors],
    #     str('u_exact'),
    #     str(save_path),
    #     np.linspace(T_initial, T_final, nt+1)
    # )

    # FOM.A.hyperelasticity_model.save_time_series(
    #     [v.impl for v in u_NN_.vectors],
    #     str('u_NN'),
    #     str(save_path),
    #     np.linspace(T_initial, T_final, nt+1)
    # )

    
if __name__ == '__main__':
    main()




