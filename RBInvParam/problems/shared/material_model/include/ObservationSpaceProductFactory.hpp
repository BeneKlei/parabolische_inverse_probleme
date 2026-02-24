#pragma once

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/dofs/dof_handler.h>

#include "ProductFactory.hpp"

using namespace dealii;

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
struct ObservationSpaceProductFactoryContext {
  const ObservationSpaceProductType  &obs_space_product_type;
  const FiniteElement<dim>           &fe;
  const DoFHandler<dim>              &dof_handler;
  const SparsityPattern              &state_sparsity_pattern;
  const size_t                       &observation_space_dim;
};

template <int dim, typename Number>
class ObservationSpaceProductFactory
{
public:
    void assemble_observation_space_product(
      const ObservationSpaceProductFactoryContext<dim, Number>& ctx,
      SparseMatrix<Number>& observation_space_product_matrix,
      SparsityPattern& observation_space_product_sp
    ) const;

    void assemble_euclid_product(
      const ObservationSpaceProductFactoryContext<dim, Number>& ctx,
      SparseMatrix<Number>& observation_space_product_matrix,
      SparsityPattern& observation_space_product_sp
    ) const;

    void assemble_product(
      const ObservationSpaceProductFactoryContext<dim, Number>& ctx,
      SparseMatrix<Number>& observation_space_product_matrix,
      SparsityPattern& observation_space_product_sp
    ) const;

private:
    ProductFactory<dim, Number> m_state_product_factory = ProductFactory<3, Number>();
};