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

#include <deal.II/base/function.h>

#include <fstream>
#include <iostream>

#include "MaterialModel.hpp"
#include "BodyForce.hpp"
#include "utils.hpp"

MaterialModel::MaterialModel(const MaterialModelConfig& config)
  : m_config(config), 
    m_fe(dealii::FE_Q<dim>(1), dim),
    m_dof_handler(m_triangulation)
{
  const double computed_nt = (m_config.T_final - m_config.T_initial) / m_config.delta_t;
  if (std::abs(computed_nt - static_cast<double>(m_config.nt)) > 1e-8) {
    throw std::runtime_error("Invalid time discretization: check T_final, T_initial, delta_t, and nt.");
  }
  
  m_param_space_dim = 0;
  m_state_space_dim = 0;

}

void MaterialModel::make_grid()
{
    Point<3> ori  = Point<3>(-0.1, -15.0, -15.0);
    Point<3> dest = Point<3>( 0.1,  15.0,  15.0);

    GridGenerator::subdivided_hyper_rectangle(
        m_triangulation, 
        m_config.spatial_resolution, 
        ori, 
        dest
    ); 

    for (const auto &face : m_triangulation.active_face_iterators())
    {
        if (face->at_boundary())
        {
            bool is_left  = true;
            bool is_right = true;

            for (unsigned int v = 0; v < GeometryInfo<3>::vertices_per_face; ++v)
            {
                if (std::fabs(face->vertex(v)[0] - ori[0]) > 1e-12)
                    is_left = false;
                if (std::fabs(face->vertex(v)[0] - dest[0]) > 1e-12)
                    is_right = false;
            }

            if (is_left)
                face->set_boundary_id(1); // left x-plane
            else if (is_right)
                face->set_boundary_id(2); // right x-plane
        }
    }
}

void MaterialModel::setup_system()
{
  m_dof_handler.clear();
  m_dof_handler.distribute_dofs(m_fe);

  m_system_matrix_sp.reinit(m_dof_handler.n_dofs(), m_dof_handler.n_dofs(), m_dof_handler.max_couplings_between_dofs());
  DoFTools::make_sparsity_pattern(m_dof_handler, m_system_matrix_sp);
  m_system_matrix_sp.compress();

  std::cout << "\t Setting up BC constraints." << std::endl;
  setup_BC_constraints();
  std::cout << "\t Setting up system matrizies." << std::endl;

  MaterialMatricesFactoryContext<3, Number> ctx {
    m_config.system_matrix_type,
    m_fe,
    m_dof_handler,
    m_BC_constraints,
    m_system_matrix_sp,
    m_config.system_matrix_hyperparameter
  };

  m_material_matrices_factory.assemble_system(
    ctx,
    m_system_matrices
  );

  std::cout << "\t Defining BodyForce." << std::endl;
  setup_body_force();
  std::cout << "\t Assembling force list." << std::endl;
  assemble_force_list();

  m_param_space_dim = m_system_matrices.get_param_space_dim();
  m_q.reinit(m_param_space_dim);
  m_state_space_dim = m_dof_handler.n_dofs();

  std::cout << "\t ---------------------- " << std::endl;
  std::cout << "\t #DoFs: " << m_state_space_dim  << std::endl;
  std::cout << "\t #Parameter: " << m_param_space_dim  << std::endl;

}

void MaterialModel::setup_BC_constraints()
{
  m_BC_constraints.clear();  
  // Functions::ZeroFunction<dim> dirichlet_bc_function(m_fe.n_components()); 
  // uint32_t boundary_id = 0;

  // VectorTools::interpolate_boundary_values(
  //   m_dof_handler, 
  //   boundary_id, 
  //   dirichlet_bc_function, 
  //   m_BC_constraints
  // );

  m_BC_constraints.close();
}

void MaterialModel::assemble_force(Vector<Number>& result, double time) 
{
  Assert(result.size() == m_dof_handler.n_dofs(),
         ExcDimensionMismatch(result.size(), m_dof_handler.n_dofs()));

  QGaussLobatto<dim> quadrature_formula(2);
  FEValues<dim> fe_values(m_fe, quadrature_formula,
                          update_values | update_quadrature_points | update_JxW_values);

  const unsigned int n_quadrature_points = quadrature_formula.size();
  const unsigned int dofs_per_cell = m_fe.dofs_per_cell;

  Vector<Number> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<Vector<double>> body_force_values(n_quadrature_points, Vector<double>(dim));

  result = 0;
  m_body_force->set_time(time);

  typename DoFHandler<dim>::active_cell_iterator cell = m_dof_handler.begin_active(), endc = m_dof_handler.end();
  for (; cell != endc; ++cell) {
    fe_values.reinit(cell);
    cell_rhs = 0;
    cell->get_dof_indices(local_dof_indices);
    m_body_force->vector_value_list(fe_values.get_quadrature_points(), body_force_values);

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
  m_system_matrices.assemble(system_matrix, m_q);
}


void MaterialModel::assemble_state_product(SparseMatrix<Number>& state_product_matrix, const StateProductType state_product_type) {
  StateProductFactoryContext<3, Number> ctx {
    state_product_type,
    m_fe,
    m_dof_handler,
    m_system_matrix_sp    
  };

  m_state_product_factory.assemble_state_product(
    ctx,
    state_product_matrix
  );
}

  

// //TODO Move all the product assemblies into ProductMatrixFactory or so
// void MaterialModel::assemble_l2_matrix(SparseMatrix<Number>& l2_matrix)
// {
//   auto l2_integrand = std::function<Number(unsigned int, unsigned int, unsigned int, const FEValues<dim>&)>(
//   [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
//     return fe.shape_value(i, q) * fe.shape_value(j, q);
//   });

//   AffineConstraints<Number> empty_BC_constraints;
//   empty_BC_constraints.clear(); 
//   empty_BC_constraints.close(); 
//   _assemble_product_matrix(l2_matrix, l2_integrand, empty_BC_constraints);
// }

// void MaterialModel::assemble_l2_0_matrix(SparseMatrix<Number>& l2_0_matrix)
// {
//   auto l2_integrand = std::function<Number(unsigned int, unsigned int, unsigned int, const FEValues<dim>&)>(
//   [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
//     return fe.shape_value(i, q) * fe.shape_value(j, q);
//   });
  
//   Functions::ZeroFunction<dim> zero_bc_function(m_fe.n_components()); 
//   uint32_t boundary_id = 0;
//   AffineConstraints<Number> zero_BC_constraints;

//   VectorTools::interpolate_boundary_values(
//     m_dof_handler, 
//     boundary_id, 
//     zero_bc_function, 
//     zero_BC_constraints
//   );

//   zero_BC_constraints.close();
//   _assemble_product_matrix(l2_0_matrix, l2_integrand, zero_BC_constraints);
// }

// void MaterialModel::assemble_h1_semi_matrix(SparseMatrix<Number>& h1_semi_matrix)
// {
//   auto h1_semi_integrand = std::function<Number(unsigned int, unsigned int, unsigned int, const FEValues<dim>&)>(
//   [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
//     return fe.shape_grad(i, q) * fe.shape_grad(j, q);
//   });
//   AffineConstraints<Number> empty_BC_constraints;
//   empty_BC_constraints.clear(); 
//   empty_BC_constraints.close();

//   _assemble_product_matrix(h1_semi_matrix, h1_semi_integrand, empty_BC_constraints);
// }

// void MaterialModel::assemble_h1_0_semi_matrix(SparseMatrix<Number>& h1_0_semi_matrix)
// {
//   auto h1_semi_integrand = std::function<Number(unsigned int, unsigned int, unsigned int, const FEValues<dim>&)>(
//   [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
//     return fe.shape_grad(i, q) * fe.shape_grad(j, q);
//   });
  
//   Functions::ZeroFunction<dim> zero_bc_function(m_fe.n_components()); 
//   uint32_t boundary_id = 0;
//   AffineConstraints<Number> zero_BC_constraints;

//   VectorTools::interpolate_boundary_values(
//     m_dof_handler, 
//     boundary_id, 
//     zero_bc_function, 
//     zero_BC_constraints
//   );

//   zero_BC_constraints.close();
//   _assemble_product_matrix(h1_0_semi_matrix, h1_semi_integrand, zero_BC_constraints);
// }

// void MaterialModel::assemble_h1_matrix(SparseMatrix<Number>& h1_matrix)
// {
//   auto h1_integrand = std::function<Number(unsigned int, unsigned int, unsigned int, const FEValues<dim>&)>(
//   [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
//     return fe.shape_grad(i, q) * fe.shape_grad(j, q) + fe.shape_grad(i, q) * fe.shape_grad(j, q);;
//   });

//   AffineConstraints<Number> empty_BC_constraints;
//   empty_BC_constraints.clear(); 
//   empty_BC_constraints.close();

//   _assemble_product_matrix(h1_matrix, h1_integrand, empty_BC_constraints);
// }

// void MaterialModel::assemble_h1_0_matrix(SparseMatrix<Number>& h1_0_matrix)
// {
//   auto h1_integrand = std::function<Number(unsigned int, unsigned int, unsigned int, const FEValues<dim>&)>(
//   [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
//     return fe.shape_grad(i, q) * fe.shape_grad(j, q) + fe.shape_grad(i, q) * fe.shape_grad(j, q);;
//   });

//   Functions::ZeroFunction<dim> zero_bc_function(m_fe.n_components()); 
//   uint32_t boundary_id = 0;
//   AffineConstraints<Number> zero_BC_constraints;

//   VectorTools::interpolate_boundary_values(
//     m_dof_handler, 
//     boundary_id, 
//     zero_bc_function, 
//     zero_BC_constraints
//   );

//   zero_BC_constraints.close();

//   _assemble_product_matrix(h1_0_matrix, h1_integrand, zero_BC_constraints);
// }

void MaterialModel::assemble_mass_matrix(SparseMatrix<Number>& mass_matrix)
{
  StateProductFactoryContext<3, Number> ctx {
    StateProductType::Mass,
    m_fe,
    m_dof_handler,
    m_system_matrix_sp    
  };

  m_state_product_factory.assemble_state_product(
    ctx,
    mass_matrix
  );
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
    ObservationOperatorType observation_operator_type)
{
    m_observation_operator_factory.assemble_observation(
      observation_operator_type,
      m_fe,
      m_dof_handler,
      m_BC_constraints,
      m_system_matrix_sp,
      operator_matrix,
      m_observation_operator_sp
  ); 
}



// void MaterialModel::assemble_euclidian_matrix(SparseMatrix<Number>& matrix)
// {
//   matrix.reinit(m_system_matrix_sp);
//   matrix = 0;

//   const auto n_dofs = m_dof_handler.n_dofs();
//   for (types::global_dof_index i = 0; i < n_dofs; ++i)
//     matrix.set(i, i, Number(1));
// };

void MaterialModel::clear_rhs_boundary_dofs(Vector<Number>& v) 
{
  m_BC_constraints.distribute(v);
}

void MaterialModel::assemble_system_matrix_derivative(
  FullMatrix<Number>& system_matrix_derivative,
  const Vector<Number>& state_DoFs)
{
    assert(
      (system_matrix_derivative.m() == m_state_space_dim) && 
      (system_matrix_derivative.n() == m_param_space_dim)
    );
    assert(m_system_matrices.m_param_space_dim == m_param_space_dim &&
       "Mismatch between system matrices count and parameter dimension");

    unsigned int offset = m_system_matrices.m_affine ? 1 : 0;
    Vector<Number> A_q_basis_u;
    //A_q_basis_u.reinit(m_state_space_dim);
    
    for (size_t i = 0; i < m_param_space_dim; i++) {
        A_q_basis_u.reinit(m_state_space_dim);
        m_system_matrices.m_matrices[i + offset].vmult(A_q_basis_u, state_DoFs);
        for (size_t j = 0; j < m_state_space_dim; j++) {
          system_matrix_derivative.set(j,i, A_q_basis_u[j]);
        }        
    }
}
void MaterialModel::assemble_bilinear_cost_matrix(
  SparseMatrix<Number>& matrix,
  const SparseMatrix<Number>& prod_C,
  const SparseMatrix<Number>& C)
{
  SparseMatrix<Number> buf;
  SparsityPattern buf_sp = utils::make_product_sparsity_AB(prod_C, C);
  buf.reinit(buf_sp);
  prod_C.mmult(buf, C, Vector<Number>(), false);
  
  SparsityPattern buf_sp_ = utils::make_product_sparsity_ATB(C, buf);
  m_bilinear_cost_sp.copy_from(buf_sp_);
  matrix.reinit(m_bilinear_cost_sp);

  C.Tmmult(matrix, buf, Vector<Number>(), false); 
}

// TODO Make a body force factory
void MaterialModel::setup_body_force() {
  switch (m_config.body_force_type)
  {
  case BodyForceType::CenterExcite:
    std::cout << "\t Using CenterExcite BodyForce" << std::endl;
    m_body_force = std::make_unique<CenterExciteBodyForce>();
    break;
  default:
    throw std::runtime_error("Unknown body force.");
  }
}