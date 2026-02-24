// =======================================
// ObservationOperatorFactory.hpp  (refactored to Option-1 style)
// =======================================
#pragma once

#include <deal.II/base/point.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <map>
#include <string>
#include <variant>
#include <vector>

#include "FESpaceContext/FESpaceContext.hpp"
#include "ProductFactory.hpp"

enum class ObservationOperatorType {
  Identity,
  Boundary,
  Sensors,
  SensorsGrid
};

using ObservationOperatorHyperparameterType =
  std::variant<bool, int, double, std::string, std::vector<double>>;
using ObservationOperatorHyperparameter =
  std::map<std::string, ObservationOperatorHyperparameterType>;

template <int dim, typename Number>
struct ObservationOperatorFactoryContext
{
  const ObservationOperatorType              &observation_operator_type;
  const FESpaceContext<dim, Number>          &space;          // <-- NEW: single source of FE data
  const ObservationOperatorHyperparameter    &hyperparameter;
};

template <int dim, typename Number>
class ObservationOperatorFactory
{
public:
  void assemble_observation(const ObservationOperatorFactoryContext<dim, Number> ctx,
                            dealii::SparseMatrix<Number>& observation_operator_matrix,
                            dealii::SparsityPattern& observation_operator_sp) const;

  void assemble_identity_observation(const ObservationOperatorFactoryContext<dim, Number> ctx,
                                     dealii::SparseMatrix<Number>& observation_operator_matrix,
                                     dealii::SparsityPattern& observation_operator_sp) const;

  void assemble_boundary_observation(const ObservationOperatorFactoryContext<dim, Number> ctx,
                                     dealii::SparseMatrix<Number>& observation_operator_matrix,
                                     dealii::SparsityPattern& observation_operator_sp) const;

  void assemble_sensors_observation(const ObservationOperatorFactoryContext<dim, Number> ctx,
                                    dealii::SparseMatrix<Number>& observation_operator_matrix,
                                    dealii::SparsityPattern& observation_operator_sp,
                                    std::vector<dealii::Point<dim>> sensor_points) const;

  // ---------------------------- utils funcs ----------------------------
  std::vector<dealii::Point<dim>> _get_sensor_edges(
      const ObservationOperatorFactoryContext<dim, Number> ctx) const;

  std::vector<dealii::Point<dim>> _get_sensor_grids(
      const ObservationOperatorFactoryContext<dim, Number> ctx) const;

private:
  ProductFactory<dim, Number> m_state_product_factory{}; // <-- no hardcoded 3
};