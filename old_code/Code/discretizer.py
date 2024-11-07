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
# The discretizer function that creates a full-order model out of the analytical problem.

import numpy as np
from pymor.discretizers.builtin import discretize_stationary_cg
from pymor.operators.constructions import LincombOperator
from pymor.parameters.functionals import ProjectionParameterFunctional, ParameterFunctional
from pymor.operators.numpy import NumpyMatrixOperator
from model import StationaryModelIP
from scipy.sparse import csr_matrix
from pymor.algorithms.preassemble import preassemble as preassemble_

def discretize_stationary_IP(analytical_problem, diameter, opt_data,
                             exact_analytical_problem,
                             energy_product_problem,
                             domain_discretizer=None,grid_type=None,
                             grid=None, boundary_info=None,
                             preassemble=False, opt_product=None,
                             compute_with_finer_mesh=False, 
                             coercivity = None):
    
    opt_data['parameters'] = analytical_problem.parameters
    if not opt_data['B_assemble']:                                               # no preassembly
        
        # discretize primal problem, but not preassemble it
        primal_fom, primal_fom_data = discretize_stationary_cg(analytical_problem,
                                                               diameter=diameter,
                                                               grid_type=grid_type,
                                                               preassemble= False
                                                               )
        

    
        # delete all parametrized oeprators and preassemble the constant ones
        true_parameterized_operator, constant_operator = extract_true_parameterized_operator(primal_fom.operator)
        matrix = constant_operator.matrix.copy()
        matrix.eliminate_zeros()
        constant_operator = NumpyMatrixOperator(matrix, range_id='STATE', source_id='STATE')
        constant_model_to_assemble = primal_fom.with_(operator = constant_operator)
        assembled_model = preassemble_(constant_model_to_assemble)
        
    elif opt_data['B_assemble']:                                                # preassembly (for large N this will take hours..)
        assembled_model, primal_fom_data = discretize_stationary_cg(analytical_problem,
                                                               diameter=diameter,
                                                               grid_type=grid_type,
                                                               preassemble= True
                                                               )
        operator = assembled_model.operator
        operators = []
        for op in operator.operators:
            matrix = op.matrix.copy()
            matrix.eliminate_zeros()
            operators.append(NumpyMatrixOperator(matrix, range_id='STATE', source_id='STATE'))
        assembled_model = assembled_model.with_(operator=operator.with_(operators=operators))
        
    # assemble the model at q_exact
    N = int(1/diameter*np.sqrt(2)) 
    refinement_factor = 2
    N_fine = N*refinement_factor
    diameter_fine = np.sqrt(2)/N_fine
    exact_assembled_fom, exact_assembled_fom_data = discretize_stationary_cg(exact_analytical_problem,
                                                           diameter= diameter_fine,#int(diameter/refinement_factor),
                                                           grid_type=grid_type,
                                                           preassemble= True
                                                           )
    
    # assemble energy product coreesponding to parameter q = 1
    q_np = np.ones((opt_data['par_dim'],))
    q_energy_product = analytical_problem.parameters.parse(q_np)
    energy_product_fom, _ = discretize_stationary_cg(energy_product_problem,
                                                           diameter=diameter,
                                                           grid_type=grid_type,
                                                           preassemble= True,
                                                           mu_energy_product = q_energy_product
                                                           )
    products = energy_product_fom.products.copy()
    
    # construct exact and noisy data
    u_exact_fine = exact_assembled_fom.solve()                                  # get exact data
    interpolate_inds = interpolate_between_grids(N_fine, refinement_factor)     
    u_exact = u_exact_fine.to_numpy()[0][interpolate_inds]                      # interpolate it on the grid             
    u_exact = assembled_model.solution_space.from_numpy(u_exact)
    np.random.seed(0)                                                           # fix random seed
    noise = np.random.rand(u_exact.dim,)                                        # get noise
    noise_L2_norm = np.sqrt(assembled_model.products['l2'].apply2(
        u_exact.space.from_numpy(noise),u_exact.space.from_numpy(noise)))[0,0]  # get l2 norm of noise
    
    if 1: # insert noise level                                                                
        noise_level = opt_data["noise_level"]
        noise_scaling = u_exact.space.from_numpy(noise_level*noise/noise_L2_norm) 
        u_noise = u_exact + noise_scaling                                       # get noisy data 
        percentage = noise_level/noise_L2_norm
        opt_data["percentage"] = percentage
        # print(f'noise percentage is {percentage}')
        # print(f'noise_level is {noise_level}')
    
    else: # insert percentage
        assert  opt_data["noise_percentage"] is not None, 'please give a valid noise percentage...'
        percentage = opt_data["noise_percentage"]
        u_exact_l2_norm = assembled_model.l2_norm(u_exact)[0]
        noise_level = u_exact_l2_norm*percentage
        opt_data["noise_level"] = noise_level
        # print(f'noise percentage is {percentage}')
        # print(f'noise_level is {noise_level}')
        noise_scaling = u_exact.space.from_numpy(noise_level*noise/noise_L2_norm)
        u_noise = u_exact + noise_scaling  
   
    # clear dirichlet dofs from u_noise
    if 'dirichlet' in opt_data['problem_type']:
        u_noise = u_noise.to_numpy()[0]
        DI = primal_fom_data["boundary_info"].dirichlet_boundaries(2)
        u_noise[DI] = 0
        u_noise = u_exact.space.from_numpy(u_noise)
    
    #opt_data["noise_level"]  =  opt_data["noise_percentage"]* u_exact_l2_norm                        
    opt_data["u_exact"] = u_exact                                               # save data in opt_data dict
    opt_data["u_noise"] = u_noise
    opt_data["noise"] = noise
    opt_data["noise_L2_norm"] = noise_L2_norm
    
    # preassemble for nonlinear output functional: bilinear, linear , constant
    opt_data["output_constant_part"] = assembled_model.l2_norm(u_noise)[0]**2       # constant part of the nonlinear output functional 
    opt_data["mass_u_noise"] = assembled_model.l2_product.apply_adjoint(u_noise)    # for linear part of output residual and constant part of adjoint rhs
    
    # get the coercivity constant and V product
    if coercivity is not None:
        opt_data["coercivity_q"] = coercivity  
    else: # choose coercivity constnat in mu scalar product
        if 'dirichlet' in opt_data['problem_type'] and 'diffusion' in opt_data['problem_type']:
            opt_data["coercivity_q"] = lambda q: abs(min(q.to_numpy()[0]))
            opt_data["V_product"] = products['h1_0_semi']
        elif 'robin' in opt_data['problem_type'] and 'diffusion' in opt_data['problem_type']:
            opt_data["coercivity_q"] = lambda q: abs(min(1,min(q.to_numpy()[0])))
            opt_data["V_product"] = products['energy']
        elif 'dirichlet' in opt_data['problem_type'] and 'reaction' in opt_data['problem_type']:
            #opt_data["coercivity_q"] = lambda q: 1 # H1 semi norm
            opt_data["coercivity_q"] = lambda q: 1#abs(min(1,min(q.to_numpy()[0])))
            opt_data["V_product"] = products['h1_0_semi']
        elif 'robin' in opt_data['problem_type'] and 'reaction' in opt_data['problem_type']:
            opt_data["coercivity_q"] = lambda q: abs(min(1,min(q.to_numpy()[0])))
            opt_data["V_product"] = products['energy']
        else:
            print('No matching problemtype given. Set coercivity constant to 1:')
            opt_data["coercivity_q"] = lambda q: 1
            opt_data["V_product"] = products['energy']
      
    # get the shape functions and their gradients
    opt_data['LagrangeShapeFunctionsGrads'] = {1: lambda X: np.array(([X[..., 1] - 1., X[..., 0] - 1.], # u links
                                                [1. - X[..., 1], - X[..., 0]], #u rechts
                                                [X[..., 1], X[..., 0]], # o rechts
                                                [-X[..., 1], 1. - X[..., 0]]))}# o links
    opt_data['LagrangeShapeFunctions'] =    {1: [lambda X: (1 - X[..., 0]) * (1 - X[..., 1]),
                     lambda X: (1 - X[..., 1]) * (X[..., 0]),
                     lambda X:     (X[..., 0]) * (X[..., 1]),
                     lambda X:     (X[..., 1]) * (1 - X[..., 0])]}
    
    # get the node to element projection matrix
    nodes_to_element_projection, cols, cols_switched = build_projection(primal_fom_data['grid'])
    # opt_data['cols'] = cols
    # opt_data['cols_switched'] = cols_switched
    primal_fom_data['nodes_to_element_projection'] = nodes_to_element_projection
    
    # create inverse problem model
    fom_IP = StationaryModelIP(assembled_model, primal_fom_data, opt_data,
                                 products = products,
                                 estimators = None,
                                 name = 'fom_IP',
                                 boundary_info=primal_fom_data['boundary_info']
                                 )
    
    return fom_IP, primal_fom_data


def extract_true_parameterized_operator(complete_operator):
    operators, coefficients = [], []
    constant_operators, constant_coefficients = [], []
    for coef, op in zip(complete_operator.coefficients, complete_operator.operators):
        assert not op.parametric, 'B operator needs to be a true LincombOperator'
        if isinstance(coef, ParameterFunctional) and coef.parametric:
            # then the operator is parametric
            assert isinstance(coef, ProjectionParameterFunctional), 'other cases are not implemented yet'
            operators.append(op)
            coefficients.append(coef)
        else:
            constant_operators.append(op)
            constant_coefficients.append(coef)
    constant_operator = LincombOperator(constant_operators, constant_coefficients).assemble()
    true_parameterized_operator = LincombOperator(operators, coefficients, name='true_parameterized_operator')
    return true_parameterized_operator, constant_operator

def build_projection(grid):
    rows = []
    cols = []
    data = []
    cols_switched = []
    nodes_per_axis_t = int(np.sqrt(len(grid.centers(0)))) # N
    nodes_per_axis_n = int(np.sqrt(len(grid.centers(2)))) # N+1
    for i in range(len(grid.centers(0))):
        j = i // nodes_per_axis_t
        entries = [i + j, i + j + 1, i + j + nodes_per_axis_n, i + j + nodes_per_axis_n + 1]
        rows.extend([i, i, i, i])
        cols.extend(entries)
        data.extend([1 / 4., 1 / 4., 1 / 4., 1 / 4.])
        # cols switched in order of shape functions (lower left, lower right, upper right, upper left)
        entries_switched = [entries[0],entries[1], entries[3], entries[2]]
        cols_switched.extend(entries_switched)  
    nodes_to_element_projection = csr_matrix((data, (rows, cols)))
    return nodes_to_element_projection, cols, cols_switched

def interpolate_between_grids(N_fine, refinement_factor):
    right = [i*(N_fine+1) for i in range(1,N_fine+2)]
    left = [i-N_fine for i in right]
    left = left[::refinement_factor]
    right = right[::refinement_factor]
    indices_coarse = []
    for i in range(len(left)):                                                 
        inds_x_axis = list(range(left[i], right[i]+1))
        indices_coarse.extend(inds_x_axis[::refinement_factor])  
    N = N_fine/refinement_factor
    assert len(indices_coarse) == (N+1)**2, 'wrong dimensions...'
    return [i-1 for i in indices_coarse]
