#pragma once

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/dofs/dof_handler.h>

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
  const SparsityPattern              &sparsity_pattern;
  const size_t                       &observation_space_dim;
};

template <int dim, typename Number>
class ObservationSpaceProductFactory
{
public:
    // void assemble_state_product(
    //     const StateProductFactoryContext<dim, Number>& ctx,
    //     SparseMatrix<Number>& state_product_matrix
    // ) const;
};