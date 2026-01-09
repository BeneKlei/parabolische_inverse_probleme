#pragma once

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h> 
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/full_matrix.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <filesystem>

#include "SystemMatrices.hpp"
#include "BodyForceFactory.hpp"
#include "ObservationOperatorFactory.hpp"
#include "StateProductFactory.hpp"
#include "ObservationSpaceProductFactory.hpp"


using namespace dealii;

typedef double Number;
// TODO make Class for this with "highlevel" pymor like interface
typedef std::vector<dealii::Vector<Number>> VectorArray;


struct MaterialModelBaseConfig {
    int nt = 50;
    double T_initial = 0.0;
    double T_final = 1.0;
    double delta_t = 1.0 / 50;
    std::vector<uint32_t> spatial_resolution = {4,30,30};
    BodyForceType body_force_type = BodyForceType::CenterExcite;
    BodyForceHyperparameter body_force_hyperparameter = {};    
};

class MaterialModel
{
public:
  static constexpr size_t dim{3};

  explicit MaterialModel(const MaterialModelBaseConfig& config);
  virtual ~MaterialModel() = default;

  void make_grid();
  void setup_system();

  void assemble_mass_matrix();
  void assemble_observation_operator_matrix(
    const ObservationOperatorType observation_operator_type,
    const ObservationOperatorHyperparameter hyperparameter
  );

  void assemble_system_operator();
  //void assemble_parameteric_matrix();
  //void assemble_system_matrix_derivative(const Vector<Number>& state_DoFs, size_t parameter_basis_idx);
  void assemble_bilinear_cost_matrix();

  // --------------------------------------------------
  
  void assemble_product_V(const StateProductType state_product_type);
  void assemble_product_H(const StateProductType state_product_type);
  void assemble_product_C(const ObservationSpaceProductType obs_space_product_type);

  // --------------------------------------------------

  void get_component_dofs(Vector<Number>& state_DoFs, size_t component_idx);  
  void clear_rhs_boundary_dofs(Vector<Number>& v);  
  void save_state(const Vector<Number>& v, const std::string save_path);
  void save_time_series(const std::vector<Vector<Number>> &v,
                        const std::string &name,
                        const std::string &save_path,
                        const std::vector<double> &times);

  // --------------------------------------------------

  size_t m_param_space_dim = 0;
  size_t m_state_space_dim = 0;
  size_t m_observation_space_dim = 0;
  //bool m_has_translation_operator = false;

  // --------------------------------------------------

  Vector<Number> m_q;
  std::vector<Vector<Number>> m_force_list;

  // --------------------------------------------------

  SparseMatrix<Number> m_mass_matrix;
  
  SparseMatrix<Number> m_system_matrix;
  // TODO Make them sparse!!
  //std::vector<FullMatrix<Number>> m_system_matrix_derivatives;


  SparseMatrix<Number> m_observation_operator;
  SparseMatrix<Number> m_bilinear_cost_operator;

  // --------------------------------------------------

  SparseMatrix<Number> m_product_V;
  SparseMatrix<Number> m_product_H;
  SparseMatrix<Number> m_product_C;

  SparseMatrix<Number> m_product_L2;
  SparseMatrix<Number> m_product_H1;

  // --------------------------------------------------
  SparsityPattern m_system_matrix_sp;
  SparsityPattern m_bilinear_cost_operator_sp;
  SparsityPattern m_observation_operator_sp;
  SparsityPattern m_obs_space_product_sp;
  
protected:
  const MaterialModelBaseConfig m_base_config;
  Triangulation<dim> m_triangulation;
  FESystem<dim> m_fe;
  DoFHandler<dim> m_dof_handler;

  ObservationOperatorFactory<dim, Number> m_observation_operator_factory = ObservationOperatorFactory<3, Number>();
  StateProductFactory<dim, Number> m_state_product_factory = StateProductFactory<3, Number>();
  ObservationSpaceProductFactory<dim, Number> m_observation_space_product_factory = ObservationSpaceProductFactory<3, Number>();
  BodyForceFactory<dim, Number> m_body_force_factory = BodyForceFactory<3, Number>();
  AffineConstraints<Number> m_BC_constraints;
  std::unique_ptr<BodyForce> m_body_force;
private:
  //SystemMatrices<dim, Number> m_system_matrices;

  
  void setup_BC_constraints();
  void setup_body_force();
  
  void assemble_force_list();
  void assemble_force(Vector<Number>& result, double time);

};


