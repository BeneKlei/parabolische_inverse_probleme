import numpy as np
import copy
from typing import Dict

def set_dims(setup: Dict, N : int = 0, nt: int = 50, q_time_dep: bool = False) -> Dict:

    par_dim = (N+1)**2
    fine_N = 2 * N

    setup['dims']['N'] = N
    setup['dims']['nt'] = nt
    setup['dims']['state_dim'] = (N+1)**2
    setup['dims']['fine_state_dim'] = (fine_N+1)**2
    setup['dims']['diameter'] = np.sqrt(2)/N
    setup['dims']['fine_diameter'] = np.sqrt(2)/fine_N
    setup['dims']['par_dim'] = par_dim

    setup['problem_parameter']['N'] = N

    T_initial = setup['model_parameter']['T_initial']
    T_final = setup['model_parameter']['T_final']
    setup['model_parameter']['delta_t'] = (T_final - T_initial) / nt
    if q_time_dep:
        setup['model_parameter']['q_circ'] = 3*np.ones((nt, par_dim))
    else:
        setup['model_parameter']['q_circ'] = 3*np.ones((1, par_dim))
    setup['model_parameter']['bounds'] = [0.001*np.ones((par_dim,)), 10e2*np.ones((par_dim,))]

    return setup


reaction_setup = {
    'dims' : {
        'N': None,
        'nt': None,
        'fine_N': None,
        'state_dim': None,
        'fine_state_dim': None,
        'diameter': None,
        'fine_diameter': None,
        'par_dim': None,
        'output_dim': 1,                                                                                                                                                                         # options to preassemble affine components or not
    },
    'problem_parameter' : {
        'N': None,
        'contrast_parameter' : 2,
        'parameter_location' : 'reaction',
        'boundary_conditions' : 'dirichlet',
        'exact_parameter' : 'Kirchner',
        'time_factor' : 'sinus',
        'T_final' : 1,
    },
    'model_parameter' : {
        'name' : 'reaction_FOM', 
        'problem_type' : None,
        'T_initial' : 0,
        'T_final' : 1,
        'delta_t' : None,
        'noise_percentage' : None,
        'noise_level' : 1e-5,
        'q_circ' : None, 
        'q_exact' : None,
        'q_time_dep' : True,
        'riesz_rep_grad' : True,
        'bounds' : None,
        'parameters' : None,
        'products' : {
            'prod_H' : 'l2',
            'prod_Q' : 'l2',
            'prod_V' : 'h1_0_semi',
            'prod_C' : 'l2',
            'bochner_prod_Q' : 'bochner_l2',
            'bochner_prod_V' : 'bochner_h1'
        },
        'observation_operator' : {
            'name' : 'identity',
        }
    }
}

diffusion_setup = {
    'dims' : {
        'N': None,
        'nt': None,
        'fine_N': None,
        'state_dim': None,
        'fine_state_dim': None,
        'diameter': None,
        'fine_diameter': None,
        'par_dim': None,
        'output_dim': 1,                                                                                                                                                                         # options to preassemble affine components or not
    },
    'problem_parameter' : {
        'N': None,
        'contrast_parameter' : 2,
        'parameter_location' : 'diffusion',
        'boundary_conditions' : 'dirichlet',
        'exact_parameter' : 'PacMan',
        'time_factor' : 'sinus',
        'T_final' : 1,
    },
    'model_parameter' : {
        'name' : 'reaction_FOM', 
        'problem_type' : None,
        'T_initial' : 0,
        'T_final' : 1,
        'delta_t' : None,
        'noise_percentage' : None,
        'noise_level' : 1e-5,
        'q_circ' : None, 
        'q_exact' : None,
        'q_time_dep' : True,
        'riesz_rep_grad' : True,
        'bounds' : None,
        'parameters' : None,
        'products' : {
            'prod_H' : 'l2',
            'prod_Q' : 'h1',
            'prod_V' : 'h1_0_semi',
            'prod_C' : 'l2',
            'bochner_prod_Q' : 'bochner_h1',
            'bochner_prod_V' : 'bochner_h1'
        },
        'observation_operator' : {
            'name' : 'identity',
        }
    }
}

FOM_optimizer_parameter = {
    'method' : 'FOM_IRGNM',
    'q_0' : None,
    'alpha_0' : 1e-5,
    'tol' : 1e-14,
    'tau' : 3.5,
    'noise_level' : None,
    'theta' : 0.4,        
    'Theta' : 1.95,
    #####################
    'i_max' : 75,
    'reg_loop_max' : 10,
    'i_max_inner' : 2,
    ####################
    'lin_solver_parms' : {
        'method' : 'gd',
        'max_iter' : 1e4,
        'tol' : 1e-11,
        'inital_step_size' : 1
    },
    'use_cached_operators' : True,
    'dump_every_nth_loop' : 5
}

TR_optimizer_parameter = {
    'method' : 'TR_IRGNM',
    'q_0' : None,
    'alpha_0' : 1e-5,
    'tol' : 1e-14,
    'tau' : 3.5,
    'noise_level' : None,
    'theta' : 0.4,
    'Theta' : 1.95,
    'tau_tilde' : 3.5,
    #####################
    'i_max' : 75,
    'reg_loop_max' : 10,
    'i_max_inner' : 10,
    'agc_armijo_max_iter' : 100,
    'TR_armijo_max_iter' : 10,
    #####################
    'lin_solver_parms' : {
        'method' : 'gd',
        'max_iter' : 1e4,
        'tol' : 1e-11,
        'inital_step_size' : 1
    },
    'use_cached_operators' : True,
    'dump_every_nth_loop' : 5,
    #####################
    'eta0' : 1e-1,
    'kappa_arm' : 1e-12,
    'beta_1' : 0.95,
    'beta_2' : 3/4,
    'beta_3' : 0.5,
}
reaction_300_setup = copy.deepcopy(reaction_setup)
reaction_300_setup = set_dims(reaction_300_setup, N = 300, q_time_dep=True)

diffusion_300_setup = copy.deepcopy(diffusion_setup)
diffusion_300_setup = set_dims(diffusion_300_setup, N = 300, q_time_dep=True)


EXPERIMENTS = {}

###################################### FOM ######################################

# setup = copy.deepcopy(reaction_300_setup)
# optimizer_parameter = copy.deepcopy(FOM_optimizer_parameter)
# optimizer_parameter['q_0'] = setup['model_parameter']['q_circ']
# optimizer_parameter['noise_level'] = setup['model_parameter']['noise_level']
# optimizer_parameter['lin_solver_parms']['lin_solver_tol'] = 1e-10
# EXPERIMENTS['reaction_FOM_300_1e-10'] = (setup, optimizer_parameter)

# setup = copy.deepcopy(reaction_300_setup)
# optimizer_parameter = copy.deepcopy(FOM_optimizer_parameter)
# optimizer_parameter['q_0'] = setup['model_parameter']['q_circ']
# optimizer_parameter['noise_level'] = setup['model_parameter']['noise_level']
# optimizer_parameter['lin_solver_parms']['lin_solver_tol'] = 5 * 1e-10
# EXPERIMENTS['reaction_FOM_300_5_1e-10'] = (setup, optimizer_parameter)

# setup = copy.deepcopy(reaction_300_setup)
# optimizer_parameter = copy.deepcopy(FOM_optimizer_parameter)
# optimizer_parameter['q_0'] = setup['model_parameter']['q_circ']
# optimizer_parameter['noise_level'] = setup['model_parameter']['noise_level']
# optimizer_parameter['lin_solver_parms']['lin_solver_tol'] = 1e-9
# EXPERIMENTS['reaction_FOM_300_1e-9'] = (setup, optimizer_parameter)




# setup = copy.deepcopy(diffusion_300_setup)
# optimizer_parameter = copy.deepcopy(FOM_optimizer_parameter)
# optimizer_parameter['q_0'] = setup['model_parameter']['q_circ']
# optimizer_parameter['noise_level'] = setup['model_parameter']['noise_level']
# optimizer_parameter['lin_solver_parms']['lin_solver_tol'] = 1e-10
# EXPERIMENTS['diffusion_FOM_300_1e-10'] = (setup, optimizer_parameter)

# setup = copy.deepcopy(diffusion_300_setup)
# optimizer_parameter = copy.deepcopy(FOM_optimizer_parameter)
# optimizer_parameter['q_0'] = setup['model_parameter']['q_circ']
# optimizer_parameter['noise_level'] = setup['model_parameter']['noise_level']
# optimizer_parameter['lin_solver_parms']['lin_solver_tol'] = 5 * 1e-10
# EXPERIMENTS['diffusion_FOM_300_5_1e-10'] = (setup, optimizer_parameter)


# setup = copy.deepcopy(diffusion_300_setup)
# optimizer_parameter = copy.deepcopy(FOM_optimizer_parameter)
# optimizer_parameter['q_0'] = setup['model_parameter']['q_circ']
# optimizer_parameter['noise_level'] = setup['model_parameter']['noise_level']
# optimizer_parameter['lin_solver_parms']['lin_solver_tol'] = 1e-9
# EXPERIMENTS['diffusion_FOM_300_1e-9'] = (setup, optimizer_parameter)


###################################### TR ######################################

setup = copy.deepcopy(reaction_300_setup)
optimizer_parameter = copy.deepcopy(TR_optimizer_parameter)
optimizer_parameter['q_0'] = setup['model_parameter']['q_circ']
optimizer_parameter['noise_level'] = setup['model_parameter']['noise_level']
optimizer_parameter['lin_solver_parms']['lin_solver_tol'] = 1e-10
EXPERIMENTS['reaction_TR_300_1e-10'] = (setup, optimizer_parameter)

setup = copy.deepcopy(reaction_300_setup)
optimizer_parameter = copy.deepcopy(TR_optimizer_parameter)
optimizer_parameter['q_0'] = setup['model_parameter']['q_circ']
optimizer_parameter['noise_level'] = setup['model_parameter']['noise_level']
optimizer_parameter['lin_solver_parms']['lin_solver_tol'] = 5 * 1e-10
EXPERIMENTS['reaction_TR_300_5_1e-10'] = (setup, optimizer_parameter)

setup = copy.deepcopy(reaction_300_setup)
optimizer_parameter = copy.deepcopy(TR_optimizer_parameter)
optimizer_parameter['q_0'] = setup['model_parameter']['q_circ']
optimizer_parameter['noise_level'] = setup['model_parameter']['noise_level']
optimizer_parameter['lin_solver_parms']['lin_solver_tol'] = 1e-9
EXPERIMENTS['reaction_TR_300_1e-9'] = (setup, optimizer_parameter)




setup = copy.deepcopy(diffusion_300_setup)
optimizer_parameter = copy.deepcopy(TR_optimizer_parameter)
optimizer_parameter['q_0'] = setup['model_parameter']['q_circ']
optimizer_parameter['noise_level'] = setup['model_parameter']['noise_level']
optimizer_parameter['lin_solver_parms']['lin_solver_tol'] = 1e-10
EXPERIMENTS['diffusion_TR_300_1e-10'] = (setup, optimizer_parameter)

setup = copy.deepcopy(diffusion_300_setup)
optimizer_parameter = copy.deepcopy(TR_optimizer_parameter)
optimizer_parameter['q_0'] = setup['model_parameter']['q_circ']
optimizer_parameter['noise_level'] = setup['model_parameter']['noise_level']
optimizer_parameter['lin_solver_parms']['lin_solver_tol'] = 5 * 1e-10
EXPERIMENTS['diffusion_TR_300_5_1e-10'] = (setup, optimizer_parameter)

setup = copy.deepcopy(diffusion_300_setup)
optimizer_parameter = copy.deepcopy(TR_optimizer_parameter)
optimizer_parameter['q_0'] = setup['model_parameter']['q_circ']
optimizer_parameter['noise_level'] = setup['model_parameter']['noise_level']
optimizer_parameter['lin_solver_parms']['lin_solver_tol'] = 1e-9
EXPERIMENTS['diffusion_TR_300_1e-9'] = (setup, optimizer_parameter)