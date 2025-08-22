#include "MaterialMatricesFactory.hpp"

template class MaterialMatricesFactory<3, double>;

template <int dim, typename Number>
void MaterialMatricesFactory<dim, Number>::assemble_system_matrix(const SystemMatrixType &system_matrix_type,
                                                                  const FiniteElement<dim> &fe,
                                                                  const DoFHandler<dim>   &dof_handler,
                                                                  const AffineConstraints<Number> &constraints,
                                                                  const SparsityPattern   &sparsity_pattern,
                                                                  const SystemMatrixHyperparameter& system_matrix_hyperparameter,
                                                                  SystemMatrices<dim, Number> &system_matrices) const
{
  switch (system_matrix_type)
  {
  case SystemMatrixType::Cosserat:
    std::cout << "\t Using Cosserat SystemMatrix" << std::endl;
    MaterialMatricesFactory::assemble_cosserat_system_matrix(
        fe,
        dof_handler,
        constraints,
        sparsity_pattern,
        system_matrix_hyperparameter,
        system_matrices
    );
    break;
  default:
    throw std::runtime_error("Unknown system matrix.");
  }
}

template <int dim, typename Number>
void MaterialMatricesFactory<dim, Number>::assemble_cosserat_system_matrix(const FiniteElement<dim> &fe,
                                                                           const DoFHandler<dim>   &dof_handler,
                                                                           const AffineConstraints<Number>  &BC_constraints,
                                                                           const SparsityPattern   &sparsity_pattern,
                                                                           const SystemMatrixHyperparameter& hooke_coeff,
                                                                           SystemMatrices<dim, Number> &system_matrices) const
{
  check_required_double_keys(hooke_coeff, {"lambda", "mu"});
  double lambda = std::get<double>(hooke_coeff.at("lambda"));
  double mu = std::get<double>(hooke_coeff.at("mu"));

  QGauss<3> quadrature_formula(2);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_gradients | update_JxW_values | update_quadrature_points | update_values); const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_quadrature_points = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Resize and initialize system matrices
  const unsigned int n_matrices = 2;
  system_matrices.m_matrices.resize(n_matrices);
  system_matrices.m_affine = false;
  system_matrices.m_param_space_dim = n_matrices;
  
  for (auto &matrix : system_matrices.m_matrices)
  {
    matrix.reinit(sparsity_pattern);
    matrix = 0;
  }

  std::vector<FullMatrix<Number>> cell_matrices(n_matrices, FullMatrix<Number>(dofs_per_cell, dofs_per_cell));

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    // Reset local matrices
    for (auto &cell_matrix : cell_matrices)
      cell_matrix = 0;

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      const unsigned int component_i = fe.system_to_component_index(i).first;

      for (unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        const unsigned int component_j = fe.system_to_component_index(j).first;

        for (unsigned int q_point = 0; q_point < n_quadrature_points; ++q_point)
        {
          const Tensor<1, dim> &grad_i = fe_values.shape_grad(i, q_point);
          const Tensor<1, dim> &grad_j = fe_values.shape_grad(j, q_point);
          const double JxW = fe_values.JxW(q_point);
        
          const double sym_term = grad_i[component_j] * grad_j[component_i] +
              ((component_i == component_j) ? grad_i * grad_j : 0.0);
          
          cell_matrices[0](i, j) += lambda * grad_i[component_i] * grad_j[component_j] * JxW;  
          cell_matrices[1](i, j) += mu * sym_term * JxW;
          
        }
      }
    }

    // Insert local matrices into global system, respecting constraints
    for (unsigned int m = 0; m < n_matrices; ++m)
    {
      BC_constraints.distribute_local_to_global(cell_matrices[m],
                                                  local_dof_indices,
                                                  system_matrices.m_matrices[m]);
    }
  }

  // Final condense to enforce constraints
  for (auto &matrix : system_matrices.m_matrices)
  {
    BC_constraints.condense(matrix);
  }
}  