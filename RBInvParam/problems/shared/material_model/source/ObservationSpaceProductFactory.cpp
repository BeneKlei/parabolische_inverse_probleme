#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include "ObservationSpaceProductFactory.hpp"

template class ObservationSpaceProductFactory<3, double>;

namespace
{
template <int dim, typename Number>
FEProductType obs_to_state_product_type(const ObservationSpaceProductType t)
{
  switch (t)
  {
    case ObservationSpaceProductType::STATE_L2:        return FEProductType::L2;
    case ObservationSpaceProductType::STATE_L2_0:      return FEProductType::L2_0;
    case ObservationSpaceProductType::STATE_H1_semi:   return FEProductType::H1_semi;
    case ObservationSpaceProductType::STATE_H1_0_semi: return FEProductType::H1_0_semi;
    case ObservationSpaceProductType::STATE_H1:        return FEProductType::H1;
    case ObservationSpaceProductType::STATE_H1_0:      return FEProductType::H1_0;
    default:
      throw std::runtime_error("obs_to_state_product_type called with non-STATE_* type.");
  }
}
} // namespace

template <int dim, typename Number>
void ObservationSpaceProductFactory<dim, Number>::assemble_observation_space_product(
    const ObservationSpaceProductFactoryContext<dim, Number>& ctx,
    dealii::SparseMatrix<Number>& observation_space_product_matrix,
    dealii::SparsityPattern& observation_space_product_sp) const
{
  switch (ctx.obs_space_product_type)
  {
    case ObservationSpaceProductType::EUCLID:
      assemble_euclid_product(ctx, observation_space_product_matrix, observation_space_product_sp);
      break;

    case ObservationSpaceProductType::STATE_L2:
    case ObservationSpaceProductType::STATE_L2_0:
    case ObservationSpaceProductType::STATE_H1_semi:
    case ObservationSpaceProductType::STATE_H1_0_semi:
    case ObservationSpaceProductType::STATE_H1:
    case ObservationSpaceProductType::STATE_H1_0:
      assemble_state_product(ctx, observation_space_product_matrix, observation_space_product_sp);
      break;

    default:
      throw std::runtime_error("Unknown ObservationSpaceProductType.");
  }
}

template <int dim, typename Number>
void ObservationSpaceProductFactory<dim, Number>::assemble_euclid_product(
    const ObservationSpaceProductFactoryContext<dim, Number>& ctx,
    dealii::SparseMatrix<Number>& observation_space_product_matrix,
    dealii::SparsityPattern& observation_space_product_sp) const
{
  const unsigned int n = static_cast<unsigned int>(ctx.observation_space_dim);

  dealii::DynamicSparsityPattern dsp(n, n);
  for (dealii::types::global_dof_index i = 0; i < n; ++i)
    dsp.add(i, i);

  observation_space_product_sp.copy_from(dsp);
  observation_space_product_matrix.reinit(observation_space_product_sp);
  observation_space_product_matrix = 0;

  for (dealii::types::global_dof_index i = 0; i < n; ++i)
    observation_space_product_matrix.set(i, i, Number(1));
}

template <int dim, typename Number>
void ObservationSpaceProductFactory<dim, Number>::assemble_state_product(
    const ObservationSpaceProductFactoryContext<dim, Number>& ctx,
    dealii::SparseMatrix<Number>& observation_space_product_matrix,
    dealii::SparsityPattern& observation_space_product_sp) const
{
  const auto& sp = ctx.space.sparsity_pattern();

  AssertThrow(ctx.observation_space_dim == sp.n_rows() &&
              ctx.observation_space_dim == sp.n_cols(),
              dealii::ExcMessage("Observation space dim != state space dim for STATE_* product."));

  const FEProductType state_ptype = obs_to_state_product_type<dim, Number>(ctx.obs_space_product_type);

  const ProductFactoryContext<dim, Number> state_product_ctx{
      state_ptype,
      ctx.space
  };

  observation_space_product_sp.copy_from(sp);
  observation_space_product_matrix.reinit(observation_space_product_sp);

  m_state_product_factory.assemble_product(state_product_ctx, observation_space_product_matrix);
}
