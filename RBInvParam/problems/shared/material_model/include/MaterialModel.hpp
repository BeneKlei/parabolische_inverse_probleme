// MaterialModel.hpp
#pragma once

// C++
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// deal.II
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

// project headers
#include "BodyForceFactory.hpp"
#include "BoundaryConditionFactory.hpp"
#include "ObservationOperatorFactory.hpp"
#include "ObservationSpaceProductFactory.hpp"

// keep this include if you haven't refactored the product factory yet
#include "ProductFactory.hpp"

#include "FESpaceContext/ParamSpaceContext.hpp"
#include "FESpaceContext/StateSpaceContext.hpp"

#include "MatrixOperator.hpp"

using Number = double;

// ======================================================
// Material Model Base Config
// ======================================================

struct MaterialModelBaseConfig
{
  static constexpr std::size_t dim{3};

  int nt = 50;
  double T_initial = 0.0;
  double T_final = 1.0;

  std::vector<double> p1 = {-0.1, -15.0, -15.0};
  std::vector<double> p2 = { 0.1,  15.0,  15.0};

  std::vector<unsigned int> param_grid_resolution = {4, 30, 30};
  std::vector<unsigned int> state_grid_resolution = {4, 30, 30};

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
  static constexpr std::size_t dim{3};
  using SparMatOp = SparseMatrixOperator<Number>;

  explicit MaterialModel(const MaterialModelBaseConfig &config);
  virtual ~MaterialModel() = default;

  // lifecycle
  void setup_system();
  virtual void setup_material_operator() = 0;

  // assembly API exposed to Python
  std::unique_ptr<SparMatOp> assemble_mass_op() const;

  std::unique_ptr<SparMatOp> assemble_observation_op(
      ObservationOperatorType observation_operator_type,
      ObservationOperatorHyperparameter hyperparameter);

  std::unique_ptr<SparMatOp> assemble_bilinear_cost_op(
      const SparMatOp &obs_op,
      const SparMatOp &product_C_op);

  std::unique_ptr<SparMatOp> assemble_state_product_op(FEProductType state_product_type) const;

  std::unique_ptr<SparMatOp> assemble_product_C_op(ObservationSpaceProductType obs_space_product_type);

  // utilities
  void clear_rhs_boundary_dofs(dealii::Vector<Number> &v);

  void save_state(const dealii::Vector<Number> &v,
                  const std::string &save_path);

  void save_time_series(const std::vector<dealii::Vector<Number>> &v,
                        const std::string &name,
                        const std::string &save_path,
                        const std::vector<double> &times);

  // contexts (views)
  const StateSpaceContext<dim, Number> &state_space_context() const { return m_state_space_context; }
  const ParamSpaceContext<dim, Number> &param_space_context() const { return m_param_space_context; }

  // flags
  const bool &q_time_dep() const { return m_q_time_dep; }
  const bool &A_affine() const { return m_A_affine; }
  const bool &A_q_linear() const { return m_A_q_linear; }

  // dimensions (kept public to minimize code churn; consider getters later)
  std::size_t m_param_dim = 0;
  std::size_t m_state_dim = 0;
  std::size_t m_observation_space_dim = 0;

  bool m_has_translation_operator = false;

  // time step size
  Number delta_t = 0.0;

  // forcing time series
  std::vector<dealii::Vector<Number>> m_force_list;

protected:
  // config
  const MaterialModelBaseConfig m_base_config;

  // ---------------------- State FE ----------------------
  dealii::Triangulation<dim> m_state_triangulation;
  dealii::FESystem<dim>      m_state_fe;
  dealii::DoFHandler<dim>    m_state_dof_handler;
  dealii::QGaussLobatto<dim> m_state_quadrature;
  dealii::MappingQ1<dim>     m_state_mapping;
  dealii::SparsityPattern    m_state_sp;

  // ---------------------- Param FE ----------------------
  dealii::Triangulation<dim>  m_param_triangulation;
  dealii::FE_Q<dim>           m_param_fe;
  dealii::DoFHandler<dim>     m_param_dof_handler;
  dealii::QGaussLobatto<dim>  m_param_quadrature;
  dealii::MappingQ1<dim>      m_param_mapping;
  dealii::SparsityPattern     m_param_sp;   // NEW: needed for param products

  dealii::AffineConstraints<Number>            m_param_constraints;
  std::vector<dealii::types::global_dof_index> m_param_free_dofs;

  // ---------------------- Constraints ----------------------
  dealii::AffineConstraints<Number> m_BC_constraints;

  // ---------------------- Contexts ----------------------
  StateSpaceContext<dim, Number> m_state_space_context;
  ParamSpaceContext<dim, Number> m_param_space_context;

  // ---------------------- Factories ---------------------
  ObservationOperatorFactory<dim, Number>      m_observation_operator_factory = ObservationOperatorFactory<3, Number>();
  ObservationSpaceProductFactory<dim, Number>  m_observation_space_product_factory = ObservationSpaceProductFactory<3, Number>();
  ProductFactory<dim, Number>             m_state_product_factory = ProductFactory<3, Number>(); // replace later with generic factory
  BodyForceFactory<dim, Number>                m_body_force_factory = BodyForceFactory<3, Number>();
  BoundaryConditionFactory<dim, Number>        m_bc_factory = BoundaryConditionFactory<3, Number>();

  // body force
  std::unique_ptr<BodyForce> m_body_force;

  // ---------------------- Cost operator sparsity ----------------------
  dealii::SparsityPattern m_bilinear_cost_operator_sp;
  dealii::SparsityPattern m_observation_operator_sp;
  dealii::SparsityPattern m_obs_space_product_sp;

  // ---------------------- Model flags ----------------------
  bool m_q_time_dep = false;
  bool m_A_affine   = true;
  bool m_A_q_linear = false;

private:
  // setup steps
  void setup_param_grid();
  void setup_state_grid();
  void setup_param_space();
  void setup_state_space();

  void setup_BC_constraints();

  // forcing
  void assemble_force_list();
  void assemble_force(dealii::Vector<Number> &result, double time);
};