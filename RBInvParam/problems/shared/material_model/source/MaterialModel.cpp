// C++
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// deal.II implementation headers
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>           // only needed if you uncomment ZeroFunction

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>   // only needed if you uncomment interpolate_boundary_values

#include "MaterialModel.hpp"
#include "utils.hpp"


MaterialModel::MaterialModel(const MaterialModelBaseConfig& config)
  : m_base_config(config) 
  , m_fe(FE_Q<dim>(1), dim)
  , m_dof_handler(m_triangulation)
  , m_param_fe(FE_Q<dim>(1))
  , m_param_dof_handler(m_param_triangulation)
  , m_param_evaluator(
      m_param_mapping,
      m_param_fe,
      update_values,
      0
    )
  , m_state_space_context(m_triangulation,
                          m_fe,
                          m_dof_handler,
                          m_state_sp, 
                          m_BC_constraints)
  , m_param_space_context(m_param_triangulation,
                          m_param_fe,
                          m_param_dof_handler,
                          m_param_mapping,
                          m_param_evaluator,
                          m_param_grid_cache,
                          m_param_constraints,
                          m_param_free_dofs)
{
  delta_t = (m_base_config.T_final - m_base_config.T_initial) / m_base_config.nt;
}

void MaterialModel::setup_param_grid()
{
    GridGenerator::subdivided_hyper_rectangle(
        m_param_triangulation, 
        m_base_config.param_grid_resolution,
        m_base_config.p1, 
        m_base_config.p2
    ); 

    m_param_grid_cache = std::make_unique<GridTools::Cache<3>>(
      m_param_triangulation, 
      m_param_mapping
    );
    
}

void MaterialModel::setup_state_grid()
{
    GridGenerator::subdivided_hyper_rectangle(
        m_triangulation, 
        m_base_config.state_grid_resolution, 
        m_base_config.p1, 
        m_base_config.p2
    ); 

    for (const auto &face : m_triangulation.active_face_iterators())
    {
        if (face->at_boundary())
        {
            bool is_left  = true;
            bool is_right = true;

            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v)
            {
                if (std::fabs(face->vertex(v)[0] - m_base_config.p1[0]) > 1e-12)
                    is_left = false;
                if (std::fabs(face->vertex(v)[0] - m_base_config.p2[0]) > 1e-12)
                    is_right = false;
            }

            if (is_left)
                face->set_boundary_id(1); // left x-plane
            else if (is_right)
                face->set_boundary_id(2); // right x-plane
        }
    }
}

void MaterialModel::setup_param_space()
{
  m_param_dof_handler.clear();
  m_param_dof_handler.distribute_dofs(m_param_fe);

  // -----------------------------------------------
  m_param_constraints.clear();

  // Map global DoFs -> physical support points
  std::vector<Point<dim>> support_points(m_param_dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(
    m_param_mapping, 
    m_param_dof_handler, 
    support_points
  );

  const Number x_min = m_base_config.p1[0];
  const Number tol = 1e-12;
  for (types::global_dof_index i = 0; i < support_points.size(); ++i)
    if (std::abs(support_points[i][0] - x_min) > tol)
    {
      m_param_constraints.add_line(i);
      m_param_constraints.set_inhomogeneity(i, Number(1.0));
    }

  m_param_constraints.close();
  m_param_dim = m_param_dof_handler.n_dofs() - m_param_constraints.n_constraints();

  // -----------------------------------------------

  m_param_free_dofs.clear();
  m_param_free_dofs.reserve(m_param_dim);

  std::vector<bool> constrained(
    m_param_dof_handler.n_dofs(), 
    false
  );
  for (types::global_dof_index i = 0; i < constrained.size(); ++i)
    constrained[i] = m_param_constraints.is_constrained(i);

  for (types::global_dof_index i = 0; i < constrained.size(); ++i)
    if (!constrained[i])
      m_param_free_dofs.push_back(i);

}

void MaterialModel::setup_state_space()
{
  m_dof_handler.clear();
  m_dof_handler.distribute_dofs(m_fe);

  m_state_sp.reinit(m_dof_handler.n_dofs(), m_dof_handler.n_dofs(), m_dof_handler.max_couplings_between_dofs());
  DoFTools::make_sparsity_pattern(m_dof_handler, m_state_sp);
  m_state_sp.compress();

  m_state_dim = m_dof_handler.n_dofs();
  
}

void MaterialModel::setup_system()
{
  std::cout << "\t Setting up grids." << std::endl;
  setup_param_grid();
  setup_state_grid();

  // --------------------------------------------------

  std::cout << "\t Setting up function spaces." << std::endl;
  setup_param_space();
  setup_state_space();
  
  std::cout << "\t ---------------------- " << std::endl;
  std::cout << "\t #State DoFs: " << m_state_dim << std::endl;
  std::cout << "\t #Parameter: " << m_param_dim << std::endl;

  // --------------------------------------------------

  std::cout << "\t Setting up BC constraints." << std::endl;
  setup_BC_constraints();
  
  // --------------------------------------------------

  std::cout << "\t Setting up material operator." << std::endl;
  setup_material_operator();

  // --------------------------------------------------

  std::cout << "\t Setting up L2 & H1 in state space." << std::endl;
  StateProductFactoryContext<3, Number> ctx_product_L2 {
    StateProductType::L2,
    m_fe,
    m_dof_handler,
    m_state_sp    
  };

  m_state_product_factory.assemble_state_product(
    ctx_product_L2,
    m_product_L2
  );

  StateProductFactoryContext<3, Number> ctx_product_H1 {
    StateProductType::H1,
    m_fe,
    m_dof_handler,
    m_state_sp    
  };

  m_state_product_factory.assemble_state_product(
    ctx_product_H1,
    m_product_H1
  );

  // --------------------------------------------------

  std::cout << "\t Assembling force list." << std::endl;

  BodyForceFactoryContext<3, Number> ctx_body_force {
    m_base_config.body_force_type,
    m_fe,
    m_dof_handler,
    m_base_config.body_force_hyperparameter
  };

  m_body_force = m_body_force_factory.assemble_body_force(
    ctx_body_force
  );

  assemble_force_list();
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

  // boundary_id = 1;

  // VectorTools::interpolate_boundary_values(
  //   m_dof_handler, 
  //   boundary_id, 
  //   dirichlet_bc_function, 
  //   m_BC_constraints
  // );

  // boundary_id = 2;

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
  m_force_list.resize(m_base_config.nt+1);

  double time = m_base_config.T_initial;
  for (uint32_t idx = 0; idx <= m_base_config.nt; idx++) {
    m_force_list[idx].reinit(m_dof_handler.n_dofs());
    assemble_force(m_force_list[idx], time);
    time += delta_t;
  }
}

void MaterialModel::assemble_product_V(const StateProductType state_product_type) {
  StateProductFactoryContext<3, Number> ctx {
    state_product_type,
    m_fe,
    m_dof_handler,
    m_state_sp    
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
    m_state_sp    
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
    m_state_sp,
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
  m_mass_matrix.reinit(m_state_sp);
  m_mass_matrix = 0;
  StateProductFactoryContext<3, Number> ctx {
    StateProductType::Mass,
    m_fe,
    m_dof_handler,
    m_state_sp    
  };

  m_state_product_factory.assemble_state_product(
    ctx,
    m_mass_matrix
  );
}

void MaterialModel::assemble_observation_operator_matrix(
  const ObservationOperatorType observation_operator_type,
  const ObservationOperatorHyperparameter hyperparameter
)
{
    ObservationOperatorFactoryContext<dim, Number> ctx {
      observation_operator_type,
      m_fe,
      m_dof_handler,
      m_BC_constraints,
      m_state_sp,
      hyperparameter,
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

// void MaterialModel::assemble_system_matrix_derivative(const Vector<Number>& state_DoFs, size_t parameter_basis_idx)
// {    
//     m_system_matrix_derivatives[parameter_basis_idx].reinit(m_state_dim, m_param_dim);
//     m_system_matrix_derivatives[parameter_basis_idx] = 0;
//     assert(m_system_matrices.m_param_dim == m_param_dim &&
//        "Mismatch between system matrices count and parameter dimension");
    
//     // m_system_matrix_derivative = 0;
//     unsigned int offset = m_system_matrices.m_affine ? 1 : 0;
//     Vector<Number> A_q_basis_u;
    
//     for (size_t i = 0; i < m_param_dim; i++) {
//         A_q_basis_u.reinit(m_state_dim);
//         m_system_matrices.m_matrices[i + offset].vmult(A_q_basis_u, state_DoFs);
//         for (size_t j = 0; j < m_state_dim; j++) {
//           m_system_matrix_derivatives[parameter_basis_idx].set(j,i, A_q_basis_u[j]);
//         }        
//     }
// }

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
	  for (unsigned int i=0;i<dim;i++)
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

// void MaterialModel::evaluate_param_values(
//   const std::vector<Number> &param,
//   const std::vector<Point<dim>> &points,  
//   std::vector<Number> &values
// ) const
// {
//   AssertThrow(
//     m_param_grid_cache.get() != nullptr, 
//     ExcMessage("m_param_grid_cache not initialized.")
//   );
//   AssertDimension(points.size(), m_param_dim);
  
//   // values.clear();
//   // values.resize(points.size());

//   // // m_full_param_buffer = Number(1.0);
  
//   // // // TODO move this into own function
//   // // for (unsigned int k = 0; k < m_param_free_dofs.size(); ++k)
//   // //   m_full_param_buffer[m_param_free_dofs[k]] = param[k];

//   // // m_param_constraints.distribute(m_full_param_buffer);

//   // for (std::size_t i = 0; i < points.size(); ++i)
//   // {
//   //   const auto &p = points[i];
//   //   m_param_evaluator.reinit(*m_param_grid_cache, m_param_dof_handler, p);
//   //   m_param_evaluator.evaluate(param, EvaluationFlags::values);
//   //   values[i] = m_param_evaluator.get_value(0);
//   // }
// }