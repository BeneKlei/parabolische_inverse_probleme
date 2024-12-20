import numpy as np

default_config_q_time_dep = {
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
    },
    'model_parameter' : {
        'T_initial' : 0,
        'T_final' : 1,
        'delta_t' : 1/50,
        'noise_percentage' : None,
        'noise_level' : 0,
        'q_circ' : 3*np.ones((50, 121)), 
        'q_exact' : None,
        'q_time_dep' : True,
        'bounds' : [0.001*np.ones((121,)), 10e2*np.ones((121,))],
        'parameters' : None
    },
    'optimizer_parameter' : None
}

default_config_q_not_time_dep = {}

CONFIGS = {
    'default_config_q_time_dep' : default_config_q_time_dep
}