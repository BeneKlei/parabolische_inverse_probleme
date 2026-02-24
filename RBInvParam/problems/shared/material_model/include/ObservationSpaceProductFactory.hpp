#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <cstddef>

#include "FESpaceContext/FESpaceContext.hpp"
#include "ProductFactory.hpp"

enum class ObservationSpaceProductType {
  EUCLID,
  STATE_L2,
  STATE_L2_0,
  STATE_H1_semi,
  STATE_H1_0_semi,
  STATE_H1,
  STATE_H1_0,
};

template <int dim, typename Number>
struct ObservationSpaceProductFactoryContext
{
  const ObservationSpaceProductType &obs_space_product_type;
  const FESpaceContext<dim, Number> &space;                 // <-- NEW
  const std::size_t                 &observation_space_dim; // typically == state dim for STATE_* cases
};

template <int dim, typename Number>
class ObservationSpaceProductFactory
{
public:
  void assemble_observation_space_product(
      const ObservationSpaceProductFactoryContext<dim, Number>& ctx,
      dealii::SparseMatrix<Number>& observation_space_product_matrix,
      dealii::SparsityPattern& observation_space_product_sp) const;

  void assemble_euclid_product(
      const ObservationSpaceProductFactoryContext<dim, Number>& ctx,
      dealii::SparseMatrix<Number>& observation_space_product_matrix,
      dealii::SparsityPattern& observation_space_product_sp) const;

  void assemble_state_product(
      const ObservationSpaceProductFactoryContext<dim, Number>& ctx,
      dealii::SparseMatrix<Number>& observation_space_product_matrix,
      dealii::SparsityPattern& observation_space_product_sp) const;

private:
  ProductFactory<dim, Number> m_product_factory{};     // <-- no hardcoded 3
};