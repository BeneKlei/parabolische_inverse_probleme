#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include<deal.II/base/function.h>

#include <fstream>
#include <iostream>

#include "MaterialModel.hpp"


MaterialModel::MaterialModel(const MaterialModelConfig& config)
  : m_config(config), 
    m_fe(dealii::FE_Q<dim>(1), dim),
    m_dof_handler(m_triangulation)
{
  const double computed_nt = (m_config.T_final - m_config.T_initial) / m_config.delta_t;
  if (std::abs(computed_nt - static_cast<double>(m_config.nt)) > 1e-8) {
    throw std::runtime_error("Invalid time discretization: check T_final, T_initial, delta_t, and nt.");
  }
}

void MaterialModel::make_grid()
{
  //std::vector<uint32_t> resolution = {10,50,50};
  //std::vector<uint32_t> resolution = {4,30,30};
  std::vector<uint32_t> resolution = {4,2,2};

  Point<3> ori = Point<3> (-0.1, -15.0, -15.0);
	Point<3> dest = Point<3> (0.1, 15.0, 15.0);

  GridGenerator::subdivided_hyper_rectangle(
    m_triangulation, 
    resolution, 
    ori, 
    dest
  );  
}

void MaterialModel::setup_system()
{
  m_dof_handler.clear();
  m_dof_handler.distribute_dofs(m_fe);

  std::cout << "\t #DoFs: " << m_dof_handler.n_dofs()  << std::endl;

  m_sparsity_pattern.reinit(m_dof_handler.n_dofs(), m_dof_handler.n_dofs(), m_dof_handler.max_couplings_between_dofs());
  DoFTools::make_sparsity_pattern(m_dof_handler, m_sparsity_pattern);
  m_sparsity_pattern.compress();

  m_q.reinit(m_config.par_dim);

  
  std::cout << "\t Setting up BC constraints." << std::endl;
  setup_BC_constraints();
  std::cout << "\t Setting system matrizies." << std::endl;
  setup_system_matricies();
  std::cout << "\t Assemble force list." << std::endl;
  assemble_force_list();
}

void MaterialModel::setup_BC_constraints()
{
  Functions::ZeroFunction<dim> dirichlet_bc_function(m_fe.n_components()); 
  uint32_t boundary_id = 0;

  VectorTools::interpolate_boundary_values(
    m_dof_handler, 
    boundary_id, 
    dirichlet_bc_function, 
    m_BC_constraints
  );

  m_BC_constraints.close();
}

void MaterialModel::setup_system_matricies() 
{
  QGauss<3> quadrature_formula(2);
  FEValues<dim> fe_values(m_fe, quadrature_formula,
                          update_gradients | update_JxW_values | update_quadrature_points | update_values);

  const unsigned int dofs_per_cell = m_fe.dofs_per_cell;
  const unsigned int n_quadrature_points = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Resize and initialize system matrices
  const unsigned int n_matrices = m_config.par_dim;
  m_system_matricies.m_matrices.resize(n_matrices);
  for (auto &matrix : m_system_matricies.m_matrices)
  {
    matrix.reinit(m_sparsity_pattern);
    matrix = 0;
  }

  std::vector<FullMatrix<Number>> cell_matrices(n_matrices, FullMatrix<Number>(dofs_per_cell, dofs_per_cell));

  for (const auto &cell : m_dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    // Reset local matrices
    for (auto &cell_matrix : cell_matrices)
      cell_matrix = 0;

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      const unsigned int component_i = m_fe.system_to_component_index(i).first;

      for (unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        const unsigned int component_j = m_fe.system_to_component_index(j).first;

        for (unsigned int q_point = 0; q_point < n_quadrature_points; ++q_point)
        {
          const Tensor<1, dim> &grad_i = fe_values.shape_grad(i, q_point);
          const Tensor<1, dim> &grad_j = fe_values.shape_grad(j, q_point);
          const double JxW = fe_values.JxW(q_point);
        
          const double sym_term = grad_i[component_j] * grad_j[component_i] +
              ((component_i == component_j) ? grad_i * grad_j : 0.0);
          
          cell_matrices[0](i, j) += grad_i[component_i] * grad_j[component_j] * JxW;  
          cell_matrices[1](i, j) += sym_term * JxW;
          
        }
      }
    }

    // Insert local matrices into global system, respecting constraints
    for (unsigned int m = 0; m < n_matrices; ++m)
    {
      m_BC_constraints.distribute_local_to_global(cell_matrices[m],
                                                  local_dof_indices,
                                                  m_system_matricies.m_matrices[m]);
    }
  }

  // Final condense to enforce constraints
  for (auto &matrix : m_system_matricies.m_matrices)
  {
    m_BC_constraints.condense(matrix);
  }
}

void MaterialModel::assemble_force(Vector<Number>& result, double time) 
{
  Assert(result.size() == m_dof_handler.n_dofs(),
         ExcDimensionMismatch(result.size(), m_dof_handler.n_dofs()));

  QGauss<dim> quadrature_formula(2);
  FEValues<dim> fe_values(m_fe, quadrature_formula,
                          update_values | update_quadrature_points | update_JxW_values);

  const unsigned int n_quadrature_points = quadrature_formula.size();
  const unsigned int dofs_per_cell = m_fe.dofs_per_cell;

  Vector<Number> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<Vector<double>> body_force_values(n_quadrature_points, Vector<double>(dim));

  result = 0;
  m_body_force.set_time(time);

  typename DoFHandler<dim>::active_cell_iterator cell = m_dof_handler.begin_active(), endc = m_dof_handler.end();
  for (; cell != endc; ++cell) {
    fe_values.reinit(cell);
    cell_rhs = 0;
    cell->get_dof_indices(local_dof_indices);
    m_body_force.vector_value_list(fe_values.get_quadrature_points(), body_force_values);

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      const unsigned int component_i = m_fe.system_to_component_index(i).first;

      for (unsigned int q_point = 0; q_point < n_quadrature_points; ++q_point)
      {
        cell_rhs(i) += fe_values.shape_value(i, q_point) *
                       body_force_values[q_point](component_i) *
                       fe_values.JxW(q_point);
      }
    }
    m_BC_constraints.distribute_local_to_global(cell_rhs,
                                                local_dof_indices,
                                                result);
  }

  m_BC_constraints.condense(result);
}

void MaterialModel::assemble_force_list()
{
  m_force_list.clear();
  m_force_list.resize(m_config.nt+1);

  double time = m_config.T_initial;
  for (uint32_t idx = 0; idx <= m_config.nt; idx++) {
    m_force_list[idx].reinit(m_dof_handler.n_dofs());
    assemble_force(m_force_list[idx], time);
    time += m_config.delta_t;
  }
}

void MaterialModel::assemble_system_matrix(SparseMatrix<Number>& system_matrix)
{
  m_system_matricies.sum(system_matrix, m_q);
}

template <typename Integrand>
void MaterialModel::_assemble_product_matrix(SparseMatrix<Number>& matrix,
                                             Integrand integrand,
                                             std::optional<std::reference_wrapper<const AffineConstraints<Number>>> constraints)
{
  QGauss<3> quadrature_formula(2);
  FEValues<dim> fe_values(m_fe, quadrature_formula,
                          update_values | update_gradients | update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = m_fe.dofs_per_cell;
  const unsigned int n_quadrature_points = quadrature_formula.size();
  
  FullMatrix<Number> cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  matrix.reinit(m_sparsity_pattern);
  matrix = 0;

  typename DoFHandler<dim>::active_cell_iterator cell = m_dof_handler.begin_active(), endc = m_dof_handler.end();
  for (; cell != endc; ++cell) {
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell->get_dof_indices(local_dof_indices);

    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
      const unsigned int component_i = m_fe.system_to_component_index(i).first;

      for (unsigned int j = 0; j < dofs_per_cell; ++j) {
        const unsigned int component_j = m_fe.system_to_component_index(j).first;
        
        if (component_i != component_j)
          continue;

        for (unsigned int q_point = 0; q_point < n_quadrature_points; ++q_point) {
          cell_matrix(i, j) += integrand(i, j, q_point, fe_values) * fe_values.JxW(q_point);
          
        }
      }
    }
    if (constraints)
      constraints->get().distribute_local_to_global(cell_matrix, local_dof_indices, matrix);
    else
      matrix.add(local_dof_indices, cell_matrix);

  }
  if (constraints)
    constraints->get().condense(matrix);
}

void MaterialModel::assemble_l2_matrix(SparseMatrix<Number>& l2_matrix)
{
  auto l2_integrand = std::function<Number(unsigned int, unsigned int, unsigned int, const FEValues<dim>&)>(
  [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
    return fe.shape_value(i, q) * fe.shape_value(j, q);
  });

  _assemble_product_matrix(l2_matrix, l2_integrand);
}

void MaterialModel::assemble_l2_0_matrix(SparseMatrix<Number>& l2_0_matrix)
{
  auto l2_integrand = std::function<Number(unsigned int, unsigned int, unsigned int, const FEValues<dim>&)>(
  [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
    return fe.shape_value(i, q) * fe.shape_value(j, q);
  });
  
  Functions::ZeroFunction<dim> zero_bc_function(m_fe.n_components()); 
  uint32_t boundary_id = 0;
  AffineConstraints<Number> zero_BC_constraints;

  VectorTools::interpolate_boundary_values(
    m_dof_handler, 
    boundary_id, 
    zero_bc_function, 
    zero_BC_constraints
  );

  zero_BC_constraints.close();
  _assemble_product_matrix(l2_0_matrix, l2_integrand, zero_BC_constraints);
}

void MaterialModel::assemble_h1_semi_matrix(SparseMatrix<Number>& h1_semi_matrix)
{
  auto h1_semi_integrand = std::function<Number(unsigned int, unsigned int, unsigned int, const FEValues<dim>&)>(
  [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
    return fe.shape_grad(i, q) * fe.shape_grad(j, q);
  });
  _assemble_product_matrix(h1_semi_matrix, h1_semi_integrand);
}

void MaterialModel::assemble_h1_0_semi_matrix(SparseMatrix<Number>& h1_0_semi_matrix)
{
  auto h1_semi_integrand = std::function<Number(unsigned int, unsigned int, unsigned int, const FEValues<dim>&)>(
  [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
    return fe.shape_grad(i, q) * fe.shape_grad(j, q);
  });
  
  Functions::ZeroFunction<dim> zero_bc_function(m_fe.n_components()); 
  uint32_t boundary_id = 0;
  AffineConstraints<Number> zero_BC_constraints;

  VectorTools::interpolate_boundary_values(
    m_dof_handler, 
    boundary_id, 
    zero_bc_function, 
    zero_BC_constraints
  );

  zero_BC_constraints.close();
  _assemble_product_matrix(h1_0_semi_matrix, h1_semi_integrand, zero_BC_constraints);
}

void MaterialModel::assemble_h1_matrix(SparseMatrix<Number>& h1_matrix)
{
  auto h1_integrand = std::function<Number(unsigned int, unsigned int, unsigned int, const FEValues<dim>&)>(
  [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
    return fe.shape_grad(i, q) * fe.shape_grad(j, q) + fe.shape_grad(i, q) * fe.shape_grad(j, q);;
  });
  _assemble_product_matrix(h1_matrix, h1_integrand);
}

void MaterialModel::assemble_h1_0_matrix(SparseMatrix<Number>& h1_0_matrix)
{
  auto h1_integrand = std::function<Number(unsigned int, unsigned int, unsigned int, const FEValues<dim>&)>(
  [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
    return fe.shape_grad(i, q) * fe.shape_grad(j, q) + fe.shape_grad(i, q) * fe.shape_grad(j, q);;
  });

  Functions::ZeroFunction<dim> zero_bc_function(m_fe.n_components()); 
  uint32_t boundary_id = 0;
  AffineConstraints<Number> zero_BC_constraints;

  VectorTools::interpolate_boundary_values(
    m_dof_handler, 
    boundary_id, 
    zero_bc_function, 
    zero_BC_constraints
  );

  zero_BC_constraints.close();

  _assemble_product_matrix(h1_0_matrix, h1_integrand, zero_BC_constraints);
}

void MaterialModel::assemble_mass_matrix(SparseMatrix<Number>& mass_matrix)
{
  auto l2_integrand = std::function<Number(unsigned int, unsigned int, unsigned int, const FEValues<dim>&)>(
  [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
    return fe.shape_value(i, q) * fe.shape_value(j, q);
  });
  _assemble_product_matrix(mass_matrix, l2_integrand, m_BC_constraints);
}

void MaterialModel::output_results(Vector<double>& solution) const
{
  DataOut<dim> data_out;
 
  data_out.attach_dof_handler(m_dof_handler);
  data_out.add_data_vector(solution, "solution");
 
  data_out.build_patches();
 
  std::ofstream output(dim == 2 ? "solution-2d.vtk" : "solution-3d.vtk");
  data_out.write_vtk(output);
}


void MaterialModel::assemble_observation_operator_matrix(
    SparseMatrix<Number>& operator_matrix, 
    std::string operator_name)
{
    if (operator_name == "identity")
        assemble_identity_observation_operator_matrix(operator_matrix);
    else
        throw std::runtime_error(
            "Unknown observation operator: " + operator_name +". Supported operator: 'identity'."
        );  
}

void MaterialModel::assemble_identity_observation_operator_matrix(SparseMatrix<Number>& matrix)
{
  matrix.reinit(m_sparsity_pattern);
  matrix = 0;

  const auto n_dofs = m_dof_handler.n_dofs();
  for (types::global_dof_index i = 0; i < n_dofs; ++i)
    matrix.set(i, i, Number(1));
};