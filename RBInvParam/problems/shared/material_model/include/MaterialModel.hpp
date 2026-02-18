#pragma once

// C++
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// deal.II: core types used in the class interface
#include <deal.II/base/point.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

// your project headers (types appear in members / signatures)
#include "BodyForceFactory.hpp"
#include "BoundaryConditionFactory.hpp"
#include "ObservationOperatorFactory.hpp"
#include "ObservationSpaceProductFactory.hpp"
#include "StateProductFactory.hpp"
#include "ParamSpaceContext.hpp"
#include "StateSpaceContext.hpp"

typedef double Number;

// ======================================================
// Material Model Base Config
// ======================================================

struct MaterialModelBaseConfig {
    static constexpr size_t dim{3};

    int nt = 50;
    double T_initial = 0.0;
    double T_final = 1.0;
    Point<dim> p1 = { -0.1, -15.0, -15.0 };
    Point<dim> p2 = {  0.1,  15.0,  15.0 };
    std::vector<unsigned int> param_grid_resolution = {4,30,30};
    std::vector<unsigned int> state_grid_resolution = {4,30,30};
    BodyForceType body_force_type = BodyForceType::CenterExcite;
    BodyForceHyperparameter body_force_hyperparameter = {};
    BoundaryConditionType BC_type = BoundaryConditionType::DirichletOnYandZ;
    BoundaryConditionHyperparameter BC_hyperparameter = {};
};

// ======================================================
// Material Model
// ======================================================

class MaterialModel
{
public:
  static constexpr size_t dim{3};

  explicit MaterialModel(const MaterialModelBaseConfig& config);
  virtual ~MaterialModel() = default;
  
  void setup_system();

  virtual void setup_material_operator() = 0;
  // TODO Add assemble_A_q etc. here to the interface

  void assemble_mass_matrix();
  void assemble_observation_operator_matrix(
    const ObservationOperatorType observation_operator_type,
    const ObservationOperatorHyperparameter hyperparameter
  );

  void assemble_system_operator();
  void assemble_bilinear_cost_matrix();

  // --------------------------------------------------
  
  // TODO Return as unique_ptr direct to python
  void assemble_product_V(const StateProductType state_product_type);
  void assemble_product_H(const StateProductType state_product_type);
  void assemble_product_C(const ObservationSpaceProductType obs_space_product_type);

  // --------------------------------------------------

  //void get_component_dofs(Vector<Number>& state_DoFs, size_t component_idx);  
  void clear_rhs_boundary_dofs(Vector<Number>& v);  
  void save_state(
    const Vector<Number>& v, 
    const std::string save_path
  );

  void save_time_series(
    const std::vector<Vector<Number>> &v,
    const std::string &name,
    const std::string &save_path,
    const std::vector<double> &times
  );

  // --------------------------------------------------

  const StateSpaceContext<dim, Number>& state_space_context() const
  {
    return m_state_space_context;
  }

  const ParamSpaceContext<dim, Number>& param_space_context() const
  {
    return m_param_space_context;
  }

  const bool& q_time_dep() const {
      return m_q_time_dep;
  }

  const bool& A_affine() const {
      return m_A_affine;
  }

  const bool& A_q_linear() const {
      return m_A_q_linear;
  }

  // --------------------------------------------------

  size_t m_param_dim = 0;
  size_t m_state_dim = 0;
  size_t m_observation_space_dim = 0;
  bool m_has_translation_operator = false;

  // --------------------------------------------------

  std::vector<Vector<Number>> m_force_list;

  // --------------------------------------------------

  SparseMatrix<Number> m_mass_matrix;  
  SparseMatrix<Number> m_system_matrix;
  SparseMatrix<Number> m_observation_operator;
  SparseMatrix<Number> m_bilinear_cost_operator;

  // --------------------------------------------------

  SparseMatrix<Number> m_product_V;
  SparseMatrix<Number> m_product_H;
  SparseMatrix<Number> m_product_C;

  SparseMatrix<Number> m_product_L2;
  SparseMatrix<Number> m_product_H1;

  // --------------------------------------------------
  SparsityPattern m_bilinear_cost_operator_sp;
  SparsityPattern m_observation_operator_sp;
  SparsityPattern m_obs_space_product_sp;

  Number delta_t; 


protected:
  const MaterialModelBaseConfig m_base_config;

  // ---------------------- State FE ----------------------
  Triangulation<dim> m_state_triangulation;
  FESystem<dim>      m_state_fe;
  DoFHandler<dim>    m_state_dof_handler;
  QGaussLobatto<dim> m_state_quadrature;
  SparsityPattern    m_state_sp;
  MappingQ1<dim>     m_state_mapping;

  // ---------------------- Param FE ----------------------
  Triangulation<dim>  m_param_triangulation;
  FE_Q<dim>           m_param_fe;
  DoFHandler<dim>     m_param_dof_handler;
  QGaussLobatto<dim>  m_param_quadrature;
  MappingQ1<dim>      m_param_mapping;

  AffineConstraints<Number>            m_param_constraints;
  std::vector<types::global_dof_index> m_param_free_dofs; // reduced index -> global DoF index

  // ---------------------- Contexts ----------------------

  StateSpaceContext<dim, Number> m_state_space_context;
  ParamSpaceContext<dim, Number> m_param_space_context;

  // ---------------------- Factories ---------------------

  ObservationOperatorFactory<dim, Number> m_observation_operator_factory = ObservationOperatorFactory<3, Number>();
  StateProductFactory<dim, Number> m_state_product_factory = StateProductFactory<3, Number>();
  ObservationSpaceProductFactory<dim, Number> m_observation_space_product_factory = ObservationSpaceProductFactory<3, Number>();
  BodyForceFactory<dim, Number> m_body_force_factory = BodyForceFactory<3, Number>();
  BoundaryConditionFactory<dim, Number> m_bc_factory = BoundaryConditionFactory<3, Number>();

  // ------------------------------------------------------

  AffineConstraints<Number> m_BC_constraints;
  std::unique_ptr<BodyForce> m_body_force;

  // ------------------------------------------------------
  
  bool m_q_time_dep = false;
  bool m_A_affine = true;
  bool m_A_q_linear = false;

private:  
  void setup_param_grid();
  void setup_state_grid();
  void setup_param_space();
  void setup_state_space();

  void setup_BC_constraints();
  void setup_body_force();
  
  void assemble_force_list();
  void assemble_force(Vector<Number>& result, double time);
};
