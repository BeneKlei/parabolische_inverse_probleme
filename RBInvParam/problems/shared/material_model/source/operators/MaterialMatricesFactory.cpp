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
  case SystemMatrixType::CosseratSpatial:
    std::cout << "\t Using CosseratSpatial SystemMatrix" << std::endl;
    MaterialMatricesFactory::assemble_cosserat_spatial_system(
        ctx,
        system_matrices
    );
    break;
  default:
    throw std::runtime_error("Unknown SystemMatrixType.");
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
void MaterialMatricesFactory<dim, Number>::assemble_cosserat_spatial_system(
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
  
  std::set<unsigned int> unique_vertices;

  for (const auto &cell : ctx.dof_handler.active_cell_iterators())
  {
      for (unsigned int v=0; v<GeometryInfo<3>::vertices_per_cell; ++v)
      {
          unique_vertices.insert(cell->vertex_index(v));
      }
  }
  
  unsigned int n_vecticies = unique_vertices.size();
  unsigned int n_matrices = 3 * n_vecticies;
  system_matrices.m_matrices.resize(n_matrices);
  system_matrices.m_affine = false;
  system_matrices.m_param_space_dim = n_matrices;
  
  for (auto &matrix : system_matrices.m_matrices)
  {
    matrix.reinit(ctx.sparsity_pattern);
    matrix = 0;
  }

  std::vector<FullMatrix<Number>> cell_matrices(3, FullMatrix<Number>(dofs_per_cell, dofs_per_cell));
  //FullMatrix<Number> cell_matrix = FullMatrix<Number>(dofs_per_cell, dofs_per_cell);

  for (const auto &cell : ctx.dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    for (unsigned int v=0; v<GeometryInfo<3>::vertices_per_cell; ++v)
    {
      unsigned int vertex_idx = cell->vertex_index(v);
      for (auto &cell_matrix : cell_matrices)
        cell_matrix = 0;

      unsigned int k_local = 0;
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
          if (ctx.fe.has_support_points())
          {
                const auto comp_i = ctx.fe.system_to_component_index(i).first;
                // optional: only take component 0 to be explicit
                if (comp_i != 0) continue;

                const auto &sp = ctx.fe.get_unit_support_points()[i];
                const auto &vp = GeometryInfo<dim>::unit_cell_vertex(v);


              if (sp.distance(vp) < 1e-12)
              {
                  k_local = i; // <-- local dof index on vertex v
                  break;
              }
          }
      }

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
            
            const double spline_value = fe_values.shape_value(k_local, q_point);
            
            cell_matrices[0](i, j) += spline_value * lambda * grad_i[component_i] * grad_j[component_j] * JxW;  
            cell_matrices[1](i, j) += spline_value * mu * sym_term * JxW;
            cell_matrices[2](i, j) += spline_value * nu * skew_term * JxW;
          }
        }
      }
      for (unsigned int m = 0; m < 3; ++m)
      {
        ctx.BC_constraints.distribute_local_to_global(cell_matrices[m],
                                                      local_dof_indices,
                                                      system_matrices.m_matrices[m * n_vecticies + vertex_idx]);
      }
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
  double lambda = std::get<double>(ctx.hyperparameter.at("lambda"));
  double mu = std::get<double>(ctx.hyperparameter.at("mu"));
  double nu = std::get<double>(ctx.hyperparameter.at("nu"));

  QGaussLobatto<3> quadrature_formula(2);
  FEValues<dim> fe_values(ctx.fe, quadrature_formula,
                          update_gradients | update_JxW_values | update_quadrature_points | update_values); 
  const unsigned int dofs_per_cell = ctx.fe.dofs_per_cell;
  const unsigned int n_quadrature_points = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<unsigned int> dofs_of_interest_on_boundary;

  for (const auto &cell : ctx.dof_handler.get_triangulation().active_cell_iterators())
  {
      for (unsigned int f = 0; f < GeometryInfo<3>::faces_per_cell; ++f)
      {
          if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 1)
          {
              for (unsigned int v = 0; v < GeometryInfo<3>::vertices_per_face; ++v)
              {
                  dofs_of_interest_on_boundary.push_back(cell->face(f)->vertex_index(v));
              }
          }
      }
  }

  std::sort(dofs_of_interest_on_boundary.begin(), dofs_of_interest_on_boundary.end());
  dofs_of_interest_on_boundary.erase(
      std::unique(dofs_of_interest_on_boundary.begin(), dofs_of_interest_on_boundary.end()),
      dofs_of_interest_on_boundary.end()
  );

  unsigned int n_matrices = dofs_of_interest_on_boundary.size();
  system_matrices.m_matrices.resize(n_matrices + 1);
  system_matrices.m_affine = true;
  system_matrices.m_param_space_dim = n_matrices;
  
  for (auto &matrix : system_matrices.m_matrices)
  {
    matrix.reinit(ctx.sparsity_pattern);
    matrix = 0;
  }

  FullMatrix<Number> cell_matrix(dofs_per_cell, dofs_per_cell);
  //unsigned int b = 0;
  unsigned int boundary_idx = 0;

  for (const auto &cell : ctx.dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    for (unsigned int v=0; v<GeometryInfo<3>::vertices_per_cell; ++v)
    {
      //for (auto &cell_matrix : cell_matrices)
      cell_matrix = 0;


      unsigned int vertex_idx = cell->vertex_index(v);
      unsigned int k_local = 0;
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
          if (ctx.fe.has_support_points())
          {
                const auto comp_i = ctx.fe.system_to_component_index(i).first;
                // optional: only take component 0 to be explicit
                if (comp_i != 0) continue;

                const auto &sp = ctx.fe.get_unit_support_points()[i];
                const auto &vp = GeometryInfo<dim>::unit_cell_vertex(v);


              if (sp.distance(vp) < 1e-12)
              {
                  k_local = i; // <-- local dof index on vertex v
                  break;
              }
          }
      }

      if (k_local != numbers::invalid_unsigned_int)
      {
          unsigned int vertex_idx = cell->vertex_index(v);
          auto it = std::lower_bound(dofs_of_interest_on_boundary.begin(),
                                     dofs_of_interest_on_boundary.end(),
                                     vertex_idx);

          if (it != dofs_of_interest_on_boundary.end() && *it == vertex_idx)
              boundary_idx = std::distance(dofs_of_interest_on_boundary.begin(), it);
          else
              boundary_idx = 0;
        }


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
            
            const double spline_value = fe_values.shape_value(k_local, q_point);
            //const double spline_value = 1.0;
            
            cell_matrix(i, j) += spline_value * lambda * grad_i[component_i] * grad_j[component_j] * JxW;  
            cell_matrix(i, j) += spline_value * mu * sym_term * JxW;
            cell_matrix(i, j) += spline_value * nu * skew_term * JxW;
          }
        }
      }
      ctx.BC_constraints.distribute_local_to_global(cell_matrix,
                                                    local_dof_indices,
                                                    system_matrices.m_matrices[boundary_idx]);
    }
  }
  // Final condense to enforce constraints
  for (auto &matrix : system_matrices.m_matrices)
  {
    ctx.BC_constraints.condense(matrix);
  }
} 