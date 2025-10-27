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

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/function.h>

#include <fstream>
#include <iostream>

#include "MaterialModel.hpp"
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

  // --------------------------------------------------

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
  m_has_translation_operator = m_system_matrices.m_affine;

  m_system_matrix_derivatives.resize(m_config.nt + 1);

  // --------------------------------------------------

  StateProductFactoryContext<3, Number> ctx_product_L2 {
    StateProductType::L2,
    m_fe,
    m_dof_handler,
    m_system_matrix_sp    
  };

  m_state_product_factory.assemble_state_product(
    ctx_product_L2,
    m_product_L2
  );

  StateProductFactoryContext<3, Number> ctx_product_H1 {
    StateProductType::H1,
    m_fe,
    m_dof_handler,
    m_system_matrix_sp    
  };

  m_state_product_factory.assemble_state_product(
    ctx_product_H1,
    m_product_H1
  );

  // --------------------------------------------------

  BodyForceFactoryContext<3, Number> ctx_body_force {
    m_config.body_force_type,
    m_fe,
    m_dof_handler,
    m_config.body_force_hyperparameter
  };


  m_body_force = m_body_force_factory.assemble_body_force(
    ctx_body_force
  );

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

void MaterialModel::get_component_dofs(Vector<Number>& state_DoFs, size_t component_idx)
{
  const FEValuesExtractors::Scalar comp(component_idx);
  const ComponentMask mask = m_fe.component_mask(comp);

  // Get all global DoF indices belonging to this component
  const IndexSet comp_dofs = DoFTools::extract_dofs(m_dof_handler, mask);

  // Zero out all other entries
  for (unsigned int i = 0; i < state_DoFs.size(); ++i)
    if (!comp_dofs.is_element(i))
      state_DoFs[i] = Number(0);
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

void MaterialModel::assemble_system_matrix()
{
  m_system_matrix.reinit(m_system_matrix_sp);
  m_system_matrix = 0;
  m_system_matrices.assemble(m_system_matrix, m_q);
}

void MaterialModel::assemble_parameteric_matrix()
{
  this->assemble_system_matrix();
  if (m_has_translation_operator)
  {
    m_system_matrix.add(-1.0, m_system_matrices.m_matrices[0]);
  }
}

void MaterialModel::assemble_product_V(const StateProductType state_product_type) {
  StateProductFactoryContext<3, Number> ctx {
    state_product_type,
    m_fe,
    m_dof_handler,
    m_system_matrix_sp    
  };

  m_state_product_factory.assemble_state_product(
    ctx,
    m_product_V
  );
}

void MaterialModel::assemble_product_H(const StateProductType state_product_type) {
  StateProductFactoryContext<3, Number> ctx {
    state_product_type,
    m_fe,
    m_dof_handler,
    m_system_matrix_sp    
  };

  m_state_product_factory.assemble_state_product(
    ctx,
    m_product_H
  );
}

void MaterialModel::assemble_product_C(const ObservationSpaceProductType obs_space_product_type) {
  ObservationSpaceProductFactoryContext<3, Number> ctx {
    obs_space_product_type,
    m_fe,
    m_dof_handler,
    m_system_matrix_sp,
    m_observation_space_dim     
  };

  m_observation_space_product_factory.assemble_observation_space_product(
    ctx,
    m_product_C,
    m_obs_space_product_sp
  );  
}

void MaterialModel::assemble_mass_matrix()
{
  m_mass_matrix.reinit(m_system_matrix_sp);
  m_mass_matrix = 0;
  StateProductFactoryContext<3, Number> ctx {
    StateProductType::Mass,
    m_fe,
    m_dof_handler,
    m_system_matrix_sp    
  };

  m_state_product_factory.assemble_state_product(
    ctx,
    m_mass_matrix
  );
}

void MaterialModel::assemble_observation_operator_matrix(ObservationOperatorType observation_operator_type)
{
    ObservationOperatorFactoryContext<dim, Number> ctx {
      observation_operator_type,
      m_fe,
      m_dof_handler,
      m_BC_constraints,
      m_system_matrix_sp,
    };
  
    m_observation_operator_factory.assemble_observation(
      ctx,
      m_observation_operator,
      m_observation_operator_sp
  ); 

  m_observation_space_dim = m_observation_operator.m();
}

void MaterialModel::clear_rhs_boundary_dofs(Vector<Number>& v) 
{
  m_BC_constraints.distribute(v);
}

void MaterialModel::assemble_system_matrix_derivative(const Vector<Number>& state_DoFs, size_t parameter_basis_idx)
{    
    m_system_matrix_derivatives[parameter_basis_idx].reinit(m_state_space_dim, m_param_space_dim);
    m_system_matrix_derivatives[parameter_basis_idx] = 0;
    assert(m_system_matrices.m_param_space_dim == m_param_space_dim &&
       "Mismatch between system matrices count and parameter dimension");
    
    // m_system_matrix_derivative = 0;
    unsigned int offset = m_system_matrices.m_affine ? 1 : 0;
    Vector<Number> A_q_basis_u;
    
    for (size_t i = 0; i < m_param_space_dim; i++) {
        A_q_basis_u.reinit(m_state_space_dim);
        m_system_matrices.m_matrices[i + offset].vmult(A_q_basis_u, state_DoFs);
        for (size_t j = 0; j < m_state_space_dim; j++) {
          m_system_matrix_derivatives[parameter_basis_idx].set(j,i, A_q_basis_u[j]);
        }        
    }
}

void MaterialModel::assemble_bilinear_cost_matrix()
{
  SparseMatrix<Number> buf;
  SparsityPattern buf_sp = utils::make_product_sparsity_AB(m_product_C, m_observation_operator);
  buf.reinit(buf_sp);
  m_product_C.mmult(buf, m_observation_operator, Vector<Number>(), false);
  
  SparsityPattern buf_sp_ = utils::make_product_sparsity_ATB(m_observation_operator, buf);
  m_bilinear_cost_operator_sp.copy_from(buf_sp_);
  m_bilinear_cost_operator.reinit(m_bilinear_cost_operator_sp);

  m_observation_operator.Tmmult(m_bilinear_cost_operator, buf, Vector<Number>(), false); 
}

// TODO Make a body force factory
// void MaterialModel::setup_body_force() {
//   switch (m_config.body_force_type)
//   {
//   case BodyForceType::CenterExcite:
//     std::cout << "\t Using CenterExcite BodyForce" << std::endl;
//     m_body_force = std::make_unique<CenterExciteBodyForce>();
//     break;
//   default:
//     throw std::runtime_error("Unknown body force.");
//   }
// }

void MaterialModel::save_state(const Vector<Number>& v, 
                               const std::string save_path)
{
    std::filesystem::path _save_path = std::filesystem::path(save_path);
    DataOut<3> data_out;
	  std::vector<std::string> solution_names;

	  solution_names.push_back("x");
	  solution_names.push_back("y");
	  solution_names.push_back("z");

	  std::vector<DataComponentInterpretation::DataComponentInterpretation> dci(3);
	  for (unsigned int i=0;i<3;i++)
	    dci[i] = DataComponentInterpretation::component_is_part_of_vector;
    
    data_out.attach_dof_handler(m_dof_handler);
    data_out.add_data_vector(v, solution_names, DataOut<3>::type_dof_data ,dci);
    data_out.add_data_vector(v, solution_names);
    data_out.build_patches();

	  std::ofstream output(_save_path);
	  data_out.write_vtk(output);
	  output.close();
}

void MaterialModel::save_time_series(const std::vector<Vector<double>> &v,
                                     const std::string &name,
                                     const std::string &save_path,
                                     const std::vector<double> &times)
{
    // ensure output directory exists
    std::filesystem::path dir = std::filesystem::path(save_path) / name;
    if (!std::filesystem::exists(dir))
        std::filesystem::create_directories(dir);

    DataOut<3> data_out;

    std::vector<std::string> solution_names = {"x", "y", "z"};
    std::vector<DataComponentInterpretation::DataComponentInterpretation> dci(3);
	  for (unsigned int i=0;i<3;i++)
	    dci[i] = DataComponentInterpretation::component_is_part_of_vector;

    // Open .pvd file to collect all timesteps
    const std::string pvd_filename = (dir / (name + ".pvd")).string();
    std::ofstream pvd(pvd_filename);
    pvd << "<?xml version=\"1.0\"?>\n";
    pvd << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    pvd << "  <Collection>\n";


    for (unsigned int t = 0; t < v.size(); ++t)
    {
        data_out.attach_dof_handler(m_dof_handler);
        data_out.add_data_vector(v[t], solution_names, DataOut<3>::type_dof_data ,dci);
        data_out.build_patches();

        // write .vtu file for this timestep
        const std::string vtu_filename = name + "_" + std::to_string(t) + ".vtu";
        std::ofstream vtu_file((dir / vtu_filename).string());
        data_out.write_vtu(vtu_file);

        // add entry to .pvd
        pvd << "    <DataSet timestep=\"" << times[t]
            << "\" group=\"\" part=\"0\" file=\"" << vtu_filename << "\"/>\n";

        data_out.clear();
    }

    pvd << "  </Collection>\n";
    pvd << "</VTKFile>\n";
}
