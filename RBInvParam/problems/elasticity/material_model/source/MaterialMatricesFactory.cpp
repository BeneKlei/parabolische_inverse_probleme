#include <cassert>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/component_mask.h>

#include "MaterialMatricesFactory.hpp"

template class MaterialMatricesFactory<3, double>;

template <int dim, typename Number>
void MaterialMatricesFactory<dim, Number>::assemble_system(
  const MaterialMatricesFactoryContext<dim, Number>& ctx,
  SystemMatrices<dim, Number> &system_matrices) const
{
  switch (ctx.system_matrix_type)
  {
  case SystemMatrixType::Cosserat:
    std::cout << "\t Using Cosserat SystemMatrix" << std::endl;
    MaterialMatricesFactory::assemble_cosserat_system(
        ctx,
        system_matrices
    );
    break;
  case SystemMatrixType::CosseratDelamination:
    std::cout << "\t Using CosseratDelamination SystemMatrix" << std::endl;
    MaterialMatricesFactory::assemble_cosserat_delamination_system(
        ctx,
        system_matrices
    );
    break;
  default:
    throw std::runtime_error("Unknown system matrix type.");
  }
}

template <int dim, typename Number>
void MaterialMatricesFactory<dim, Number>::assemble_cosserat_system(
  const MaterialMatricesFactoryContext<dim, Number>& ctx,
  SystemMatrices<dim, Number> &system_matrices) const
{
  check_required_keys<double>(ctx.hyperparameter, {"lambda", "mu", "nu"});
  double lambda = std::get<double>(ctx.hyperparameter.at("lambda"));
  double mu = std::get<double>(ctx.hyperparameter.at("mu"));
  double nu = std::get<double>(ctx.hyperparameter.at("nu"));

  QGaussLobatto<3> quadrature_formula(2);
  FEValues<dim> fe_values(ctx.fe, quadrature_formula,
                          update_gradients | update_JxW_values | update_quadrature_points | update_values); 
  const unsigned int dofs_per_cell = ctx.fe.dofs_per_cell;
  const unsigned int n_quadrature_points = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Resize and initialize system matrices
  const unsigned int n_matrices = 3;
  system_matrices.m_matrices.resize(n_matrices);
  system_matrices.m_affine = false;
  system_matrices.m_param_space_dim = n_matrices;
  
  for (auto &matrix : system_matrices.m_matrices)
  {
    matrix.reinit(ctx.sparsity_pattern);
    matrix = 0;
  }

  std::vector<FullMatrix<Number>> cell_matrices(n_matrices, FullMatrix<Number>(dofs_per_cell, dofs_per_cell));
  for (const auto &cell : ctx.dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    // Reset local matrices
    for (auto &cell_matrix : cell_matrices)
      cell_matrix = 0;

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      const unsigned int component_i = ctx.fe.system_to_component_index(i).first;

      for (unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        const unsigned int component_j = ctx.fe.system_to_component_index(j).first;

        for (unsigned int q_point = 0; q_point < n_quadrature_points; ++q_point)
        {
          const Tensor<1, dim> &grad_i = fe_values.shape_grad(i, q_point);
          const Tensor<1, dim> &grad_j = fe_values.shape_grad(j, q_point);
          const double JxW = fe_values.JxW(q_point);
        
          const double sym_term = grad_i[component_j] * grad_j[component_i] +
              ((component_i == component_j) ? grad_i * grad_j : 0.0);
          const double skew_term = grad_i[component_j] * grad_j[component_i]
                       - ((component_i == component_j) ? grad_i * grad_j : 0.0);
          
          cell_matrices[0](i, j) += lambda * grad_i[component_i] * grad_j[component_j] * JxW;  
          cell_matrices[1](i, j) += mu * sym_term * JxW;
          cell_matrices[2](i, j) += nu * skew_term * JxW;
        }
      }
    }

    // Insert local matrices into global system, respecting constraints
    for (unsigned int m = 0; m < n_matrices; ++m)
    {
      ctx.BC_constraints.distribute_local_to_global(cell_matrices[m],
                                                local_dof_indices,
                                                system_matrices.m_matrices[m]);
    }
  }
  // Final condense to enforce constraints
  for (auto &matrix : system_matrices.m_matrices)
  {
    ctx.BC_constraints.condense(matrix);
  }
} 


template <int dim, typename Number>
void MaterialMatricesFactory<dim, Number>::assemble_cosserat_delamination_system(
  const MaterialMatricesFactoryContext<dim, Number>& ctx,
  SystemMatrices<dim, Number> &system_matrices) const
{
  check_required_keys<double>(ctx.hyperparameter, {"lambda", "mu", "nu"});
  check_required_keys<std::string>(ctx.hyperparameter, {"surface"});
  double lambda = std::get<double>(ctx.hyperparameter.at("lambda"));
  double mu = std::get<double>(ctx.hyperparameter.at("mu"));
  double nu = std::get<double>(ctx.hyperparameter.at("nu"));
  std::string surface = std::get<std::string>(ctx.hyperparameter.at("surface"));

  if (!(surface == "left")) {
    throw std::runtime_error("A model for delamination at surface " + surface + "is not implemented. Options are ['left'].");
  }

  QGaussLobatto<3> quadrature_formula(2);
  FEValues<dim> fe_values(ctx.fe, quadrature_formula,
                          update_gradients | update_JxW_values | update_quadrature_points | update_values); 
  const unsigned int dofs_per_cell = ctx.fe.dofs_per_cell;
  const unsigned int n_quadrature_points = quadrature_formula.size();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // --- NEW: collect boundary vertices instead of boundary dofs ---
  std::set<unsigned int> boundary_vertex_indices;
  for (const auto &cell : ctx.dof_handler.active_cell_iterators())
    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 1)
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v)
          boundary_vertex_indices.insert(cell->face(f)->vertex_index(v));

  const unsigned int n_bdry_vertices = boundary_vertex_indices.size();

  const unsigned int n_matrices = 1 + n_bdry_vertices;
  system_matrices.m_matrices.resize(n_matrices);
  system_matrices.m_affine = true;
  system_matrices.m_param_space_dim = n_bdry_vertices; // one per vertex

  for (auto &A : system_matrices.m_matrices)
  {
    A.reinit(ctx.sparsity_pattern);
    A = 0.0;
  }

  unsigned int slot = 1;
  FullMatrix<Number> cell_matrix = FullMatrix<Number>(dofs_per_cell, dofs_per_cell);

  for (unsigned int v : boundary_vertex_indices)
  {
    const unsigned int target_matrix = slot++;

    // Collect global DoFs at this vertex (one per component)
    std::vector<types::global_dof_index> vertex_dofs;
    bool found = false;
    for (const auto &cell : ctx.dof_handler.active_cell_iterators())
    {
      for (unsigned int vv = 0; vv < GeometryInfo<dim>::vertices_per_cell; ++vv)
      {
        if (cell->vertex_index(vv) == v)
        {
          for (unsigned int c = 0; c < ctx.fe.n_components(); ++c)
            vertex_dofs.push_back(cell->vertex_dof_index(vv, c));
          found = true;
          break;
        }
      }
      if (found) break; // stop once we found a cell containing this vertex
    }

    for (const auto &cell : ctx.dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(local_dof_indices);
      // check if this cell contains the vertex
      bool contains_vertex_dof = false;
      unsigned int k_local = numbers::invalid_unsigned_int;

      for (unsigned int c = 0; c < ctx.fe.n_components(); ++c)
      {
        auto it = std::find(local_dof_indices.begin(),
                            local_dof_indices.end(),
                            vertex_dofs[c]);
        if (it != local_dof_indices.end())
        {
          contains_vertex_dof = true;
          k_local = static_cast<unsigned int>(std::distance(local_dof_indices.begin(), it));
          break; // we just need one local index for the shape_value
        }
      }

      if (k_local == numbers::invalid_unsigned_int)
        continue; // cell doesn't touch this vertex

      fe_values.reinit(cell);
      cell->get_dof_indices(local_dof_indices);
      cell_matrix = 0;

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const unsigned int component_i = ctx.fe.system_to_component_index(i).first;

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          const unsigned int component_j = ctx.fe.system_to_component_index(j).first;

          for (unsigned int q_point = 0; q_point < n_quadrature_points; ++q_point)
          {
            const Tensor<1, dim> &grad_i = fe_values.shape_grad(i, q_point);
            const Tensor<1, dim> &grad_j = fe_values.shape_grad(j, q_point);
            const double JxW = fe_values.JxW(q_point);
          
            const double sym_term = grad_i[component_j] * grad_j[component_i] +
                ((component_i == component_j) ? grad_i * grad_j : 0.0);
            const double skew_term = grad_i[component_j] * grad_j[component_i]
                        - ((component_i == component_j) ? grad_i * grad_j : 0.0);
            
            // IMPORTANT: use cell-local index for the weighting function
            const double spline_value = fe_values.shape_value(k_local, q_point);

            cell_matrix(i, j) += spline_value * lambda * grad_i[component_i] * grad_j[component_j] * JxW;  
            cell_matrix(i, j) += spline_value * mu * sym_term * JxW;
            cell_matrix(i, j) += spline_value * nu * skew_term * JxW;
          }
        }
      }
      ctx.BC_constraints.distribute_local_to_global(
        cell_matrix, local_dof_indices, system_matrices.m_matrices[target_matrix]);
    }
  }

  for (auto &matrix : system_matrices.m_matrices)
  {
    ctx.BC_constraints.condense(matrix);
  }
} 