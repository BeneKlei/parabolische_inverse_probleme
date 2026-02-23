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


// TODO Move the FE mechanic to the contexts
MaterialModel::MaterialModel(const MaterialModelBaseConfig& config)
  : m_base_config(config) 
  , m_state_fe(FE_Q<dim>(1), dim)
  , m_state_dof_handler(m_state_triangulation)
  , m_state_quadrature(QGaussLobatto<dim>(2))
  , m_state_space_context(m_state_triangulation,
                          m_state_fe,
                          m_state_dof_handler,
                          m_state_mapping,
                          m_state_quadrature,
                          m_state_sp, 
                          m_BC_constraints,
                          m_base_config.state_grid_resolution, 
                          utils::vec_to_point<dim>(m_base_config.p1), 
                          utils::vec_to_point<dim>(m_base_config.p2))
  , m_param_fe(FE_Q<dim>(1))
  , m_param_dof_handler(m_param_triangulation)                          
  , m_param_quadrature(QGaussLobatto<dim>(2))
  , m_param_space_context(m_state_triangulation, // For the moment this must coincide for state and param space
                          m_param_fe,
                          m_param_dof_handler,
                          m_state_mapping,  // For the moment this must coincide for state and param space
                          m_state_quadrature,  // For the moment this must coincide for state and param space
                          //m_param_rpe,
                          m_param_constraints,
                          m_param_free_dofs,
                          m_base_config.state_grid_resolution, 
                          utils::vec_to_point<dim>(m_base_config.p1), 
                          utils::vec_to_point<dim>(m_base_config.p2))
{
  delta_t = (m_base_config.T_final - m_base_config.T_initial) / m_base_config.nt;
}

void MaterialModel::setup_param_grid()
{ 
    GridGenerator::subdivided_hyper_rectangle(
        m_param_triangulation, 
        m_base_config.param_grid_resolution,
        utils::vec_to_point<dim>(m_base_config.p1), 
        utils::vec_to_point<dim>(m_base_config.p2)
    ); 
}

void MaterialModel::setup_state_grid()
{
  GridGenerator::subdivided_hyper_rectangle(
      m_state_triangulation,
      m_base_config.state_grid_resolution,
      utils::vec_to_point<dim>(m_base_config.p1),
      utils::vec_to_point<dim>(m_base_config.p2));

  const double tol = 1e-12;

  for (const auto &face : m_state_triangulation.active_face_iterators())
  {
    if (!face->at_boundary())
      continue;

    bool x_min = true, x_max = true;
    bool y_min = true, y_max = true;
    bool z_min = true, z_max = true;

    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v)
    {
      const auto &P = face->vertex(v);

      if (std::fabs(P[0] - m_base_config.p1[0]) > tol) x_min = false;
      if (std::fabs(P[0] - m_base_config.p2[0]) > tol) x_max = false;

      if (std::fabs(P[1] - m_base_config.p1[1]) > tol) y_min = false;
      if (std::fabs(P[1] - m_base_config.p2[1]) > tol) y_max = false;

      if (std::fabs(P[2] - m_base_config.p1[2]) > tol) z_min = false;
      if (std::fabs(P[2] - m_base_config.p2[2]) > tol) z_max = false;
    }

    // pick your own ID convention; just be consistent:
    if      (x_min) face->set_boundary_id(1); // x = p1[0]
    else if (x_max) face->set_boundary_id(2); // x = p2[0]
    else if (y_min) face->set_boundary_id(3); // y = p1[1]
    else if (y_max) face->set_boundary_id(4); // y = p2[1]
    else if (z_min) face->set_boundary_id(5); // z = p1[2]
    else if (z_max) face->set_boundary_id(6); // z = p2[2]
  }
}


void MaterialModel::setup_param_space()
{
  m_param_dof_handler.clear();
  m_param_dof_handler.reinit(m_param_triangulation);
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

  // TODO The computation to the contexts, maybe also the grids.
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
  m_state_dof_handler.clear();
  m_state_dof_handler.distribute_dofs(m_state_fe);

  m_state_sp.reinit(m_state_dof_handler.n_dofs(), m_state_dof_handler.n_dofs(), m_state_dof_handler.max_couplings_between_dofs());
  DoFTools::make_sparsity_pattern(m_state_dof_handler, m_state_sp);
  m_state_sp.compress();

  m_state_dim = m_state_dof_handler.n_dofs();
  
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

  std::cout << "\t Precomputing for State Context ." << std::endl;
  m_state_space_context.pre_compute();

  // --------------------------------------------------

  std::cout << "\t Setting up material operator." << std::endl;
  setup_material_operator();

  // --------------------------------------------------

  // std::cout << "\t Setting up L2 & H1 in state space." << std::endl;
  // StateProductFactoryContext<3, Number> ctx_product_L2 {
  //   StateProductType::L2,
  //   m_state_fe,
  //   m_state_dof_handler,
  //   m_state_sp    
  // };

  // m_state_product_factory.assemble_state_product(
  //   ctx_product_L2,
  //   m_product_L2
  // );

  // StateProductFactoryContext<3, Number> ctx_product_H1 {
  //   StateProductType::H1,
  //   m_state_fe,
  //   m_state_dof_handler,
  //   m_state_sp    
  // };

  // m_state_product_factory.assemble_state_product(
  //   ctx_product_H1,
  //   m_product_H1
  // );

  // --------------------------------------------------

  std::cout << "\t Assembling force list." << std::endl;

  BodyForceFactoryContext<3, Number> ctx_body_force {
    m_base_config.body_force_type,
    m_state_fe,
    m_state_dof_handler,
    m_base_config.body_force_hyperparameter
  };

  m_body_force = m_body_force_factory.assemble_body_force(
    ctx_body_force
  );

  assemble_force_list();
}

void MaterialModel::setup_BC_constraints()
{
  m_bc_factory.assemble_constraints(
    m_state_space_context, 
    m_base_config.BC_type,
    m_base_config.BC_hyperparameter,
    m_BC_constraints
  );
}


// void MaterialModel::get_component_dofs(Vector<Number>& state_DoFs, size_t component_idx)
// {
//   const FEValuesExtractors::Scalar comp(component_idx);
//   const ComponentMask mask = m_state_fe.component_mask(comp);

//   // Get all global DoF indices belonging to this component
//   const IndexSet comp_dofs = DoFTools::extract_dofs(m_state_dof_handler, mask);

//   // Zero out all other entries
//   for (unsigned int i = 0; i < state_DoFs.size(); ++i)
//     if (!comp_dofs.is_element(i))
//       state_DoFs[i] = Number(0);
// }

void MaterialModel::assemble_force(Vector<Number>& result, double time) 
{
  Assert(result.size() == m_state_dof_handler.n_dofs(),
         ExcDimensionMismatch(result.size(), m_state_dof_handler.n_dofs()));

  QGaussLobatto<dim> quadrature_formula(2);
  FEValues<dim> fe_values(m_state_fe, quadrature_formula,
                          update_values | update_quadrature_points | update_JxW_values);

  const unsigned int n_quadrature_points = quadrature_formula.size();
  const unsigned int dofs_per_cell = m_state_fe.dofs_per_cell;

  Vector<Number> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<Vector<double>> body_force_values(n_quadrature_points, Vector<double>(dim));

  result = 0;
  m_body_force->set_time(time);

  typename DoFHandler<dim>::active_cell_iterator cell = m_state_dof_handler.begin_active(), endc = m_state_dof_handler.end();
  for (; cell != endc; ++cell) {
    fe_values.reinit(cell);
    cell_rhs = 0;
    cell->get_dof_indices(local_dof_indices);
    m_body_force->vector_value_list(fe_values.get_quadrature_points(), body_force_values);

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      const unsigned int component_i = m_state_fe.system_to_component_index(i).first;
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
    m_force_list[idx].reinit(m_state_dof_handler.n_dofs());
    assemble_force(m_force_list[idx], time);
    time += delta_t;
  }
}

// void MaterialModel::assemble_product_V(const StateProductType state_product_type) {
//   StateProductFactoryContext<3, Number> ctx {
//     state_product_type,
//     m_state_fe,
//     m_state_dof_handler,
//     m_state_sp    
//   };

//   m_state_product_factory.assemble_state_product(
//     ctx,
//     m_product_V
//   );
// }

// void MaterialModel::assemble_product_H(const StateProductType state_product_type) {
//   StateProductFactoryContext<3, Number> ctx {
//     state_product_type,
//     m_state_fe,
//     m_state_dof_handler,
//     m_state_sp    
//   };

//   m_state_product_factory.assemble_state_product(
//     ctx,
//     m_product_H
//   );
// }

std::unique_ptr<MaterialModel::SparMatOp>
MaterialModel::assemble_state_product_op(
  const StateProductType state_product_type
) const 
{
  SparseMatrix<Number> product_mat;

  const StateProductFactoryContext<3, Number> ctx {
    state_product_type,
    m_state_fe,
    m_state_dof_handler,
    m_state_sp    
  };

  m_state_product_factory.assemble_state_product(
    ctx,
    product_mat
  );

  return std::make_unique<MaterialModel::SparMatOp>(
    std::move(product_mat)
  );
}


std::unique_ptr<MaterialModel::SparMatOp> 
MaterialModel::assemble_product_C_op(
  const ObservationSpaceProductType obs_space_product_type
) 
{

  SparseMatrix<Number> prod_C_mat;

  ObservationSpaceProductFactoryContext<3, Number> ctx {
    obs_space_product_type,
    m_state_fe,
    m_state_dof_handler,
    m_state_sp,
    m_observation_space_dim     
  };

  m_observation_space_product_factory.assemble_observation_space_product(
    ctx,
    prod_C_mat,
    m_obs_space_product_sp
  );  

  return std::make_unique<MaterialModel::SparMatOp>(
      std::move(prod_C_mat)
  );
}

std::unique_ptr<MaterialModel::SparMatOp> 
MaterialModel::assemble_mass_op() const
{
  return assemble_state_product_op(StateProductType::Mass);

  // m_mass_matrix.reinit(m_state_sp);
  // m_mass_matrix = 0;
  // StateProductFactoryContext<3, Number> ctx {
  //   StateProductType::Mass,
  //   m_state_fe,
  //   m_state_dof_handler,
  //   m_state_sp    
  // };

  // m_state_product_factory.assemble_state_product(
  //   ctx,
  //   m_mass_matrix
  // );
}

std::unique_ptr<MaterialModel::SparMatOp> 
MaterialModel::assemble_observation_op(
  const ObservationOperatorType observation_operator_type,
  const ObservationOperatorHyperparameter hyperparameter
)
{
    SparseMatrix<Number> obs_op_mat;

    ObservationOperatorFactoryContext<dim, Number> ctx {
      observation_operator_type,
      m_state_fe,
      m_state_dof_handler,
      m_BC_constraints,
      m_state_sp,
      hyperparameter,
    };
  
    m_observation_operator_factory.assemble_observation(
      ctx,
      obs_op_mat,
      m_observation_operator_sp
  ); 

  m_observation_space_dim = obs_op_mat.m();

  return std::make_unique<MaterialModel::SparMatOp>(
      std::move(obs_op_mat)
  );


}

void MaterialModel::clear_rhs_boundary_dofs(Vector<Number>& v) 
{
  m_BC_constraints.distribute(v);
}

// void MaterialModel::assemble_bilinear_cost_matrix()
// {
//   SparseMatrix<Number> buf;
//   SparsityPattern buf_sp = utils::make_product_sparsity_AB(m_product_C, m_observation_operator);
//   buf.reinit(buf_sp);
//   m_product_C.mmult(buf, m_observation_operator, Vector<Number>(), false);
  
//   SparsityPattern buf_sp_ = utils::make_product_sparsity_ATB(m_observation_operator, buf);
//   m_bilinear_cost_operator_sp.copy_from(buf_sp_);
//   m_bilinear_cost_operator.reinit(m_bilinear_cost_operator_sp);

//   m_observation_operator.Tmmult(m_bilinear_cost_operator, buf, Vector<Number>(), false); 
// }


std::unique_ptr<MaterialModel::SparMatOp> MaterialModel::assemble_bilinear_cost_op(
  const SparMatOp& obs_op,
  const SparMatOp& product_C_op
) 
{
  const SparseMatrix<Number>& obs_op_mat = obs_op.get_matrix();
  const SparseMatrix<Number>& product_C_mat = product_C_op.get_matrix();

  SparseMatrix<Number> bilinear_cost_operator;
  SparseMatrix<Number> buf;

  SparsityPattern buf_sp = utils::make_product_sparsity_AB(
    product_C_mat, 
    obs_op_mat
  );
  
  buf.reinit(buf_sp);
  product_C_mat.mmult(buf, obs_op_mat, Vector<Number>(), false);


  SparsityPattern buf_sp_ = utils::make_product_sparsity_ATB(obs_op_mat, buf);
  m_bilinear_cost_operator_sp.copy_from(buf_sp_);
  bilinear_cost_operator.reinit(m_bilinear_cost_operator_sp);

  obs_op_mat.Tmmult(bilinear_cost_operator, buf, Vector<Number>(), false); 
  
  return std::make_unique<MaterialModel::SparMatOp>(
      std::move(bilinear_cost_operator)
  );
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
    
    data_out.attach_dof_handler(m_state_dof_handler);
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
        data_out.attach_dof_handler(m_state_dof_handler);
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
