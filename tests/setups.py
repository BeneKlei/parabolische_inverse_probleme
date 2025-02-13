import numpy as np

default_setup_q_time_dep = {
    'dims' : {
        'N': 10,
        'nt': 50,
        'fine_N': 20,
        'state_dim': 121,
        'fine_state_dim': 441,
        'diameter': np.sqrt(2)/10,
        'fine_diameter': np.sqrt(2)/20,
        'par_dim': 121,
        'output_dim': 1
    },
    'problem_parameter' : {
        'N' : 10,
        'parameter_location' : 'reaction',
        'boundary_conditions' : 'dirichlet',
        'exact_parameter' : 'Kirchner',
        'T_final' : 1
    },
    'model_parameter' : {
        'name' : 'reaction_FOM',
        'T_initial' : 0,
        'T_final' : 1,
        'delta_t' : 1/50,
        'noise_percentage' : None,
        'noise_level' : 0,
        'q_circ' : 3*np.ones((50, 121)), 
        'q_exact' : None,
        'q_time_dep' : True,
        'riesz_rep_grad' : True,
        'bounds' : [0.001*np.ones((121,)), 10e2*np.ones((121,))],
        'parameters' : None,
        'products' : {
            'prod_H' : 'l2',
            'prod_Q' : 'l2',
            'prod_V' : 'h1',
            'prod_C' : 'l2',
            'bochner_prod_Q' : 'bochner_l2',
            'bochner_prod_V' : 'bochner_h1'
        }
    },
    'optimizer_parameter' : None
}

default_setup_q_non_time_dep = {
    'dims' : {
        'N': 10,
        'nt': 50,
        'fine_N': 20,
        'state_dim': 121,
        'fine_state_dim': 441,
        'diameter': np.sqrt(2)/10,
        'fine_diameter': np.sqrt(2)/20,
        'par_dim': 121,
        'output_dim': 1
    },
    'problem_parameter' : {
        'N' : 10,
        'parameter_location' : 'reaction',
        'boundary_conditions' : 'dirichlet',
        'exact_parameter' : 'Kirchner',
        'T_final' : 1
    },
    'model_parameter' : {
        'name' : 'reaction_FOM',
        'T_initial' : 0,
        'T_final' : 1,
        'delta_t' : 1/50,
        'noise_percentage' : None,
        'noise_level' : 0,
        'q_circ' : 3*np.ones((1, 121)), 
        'q_exact' : None,
        'q_time_dep' : False,
        'riesz_rep_grad' : True,
        'bounds' : [0.001*np.ones((121,)), 10e2*np.ones((121,))],
        'parameters' : None,
        'products' : {
            'prod_H' : 'l2',
            'prod_Q' : 'l2',
            'prod_V' : 'h1',
            'prod_C' : 'l2',
            'bochner_prod_Q' : 'bochner_l2',
            'bochner_prod_V' : 'bochner_h1'
        }
    },
    'optimizer_parameter' : None
}

diffusion_setup_q_time_dep = {
    'dims' : {
        'N': 10,
        'nt': 50,
        'fine_N': 20,
        'state_dim': 121,
        'fine_state_dim': 441,
        'diameter': np.sqrt(2)/10,
        'fine_diameter': np.sqrt(2)/20,
        'par_dim': 121,
        'output_dim': 1
    },
    'problem_parameter' : {
        'N' : 10,
        'parameter_location' : 'diffusion',
        'boundary_conditions' : 'dirichlet',
        'exact_parameter' : 'PacMan',
        'T_final' : 1
    },
    'model_parameter' : {
        'name' : 'reaction_FOM',
        'T_initial' : 0,
        'T_final' : 1,
        'delta_t' : 1/50,
        'noise_percentage' : None,
        'noise_level' : 0,
        'q_circ' : 3*np.ones((50, 121)), 
        'q_exact' : None,
        'q_time_dep' : True,
        'riesz_rep_grad' : True,
        'bounds' : [0.001*np.ones((121,)), 10e2*np.ones((121,))],
        'parameters' : None,
        'products' : {
            'prod_H' : 'l2',
            'prod_Q' : 'l2',
            'prod_V' : 'h1',
            'prod_C' : 'l2',
            'bochner_prod_Q' : 'bochner_l2',
            'bochner_prod_V' : 'bochner_h1'
        }
    },
    'optimizer_parameter' : None
}


SETUPS = {
    'default_setup_q_time_dep' : default_setup_q_time_dep,
    'default_setup_q_non_time_dep' : default_setup_q_non_time_dep,
    #'diffusion_setup_q_time_dep' : diffusion_setup_q_time_dep
}