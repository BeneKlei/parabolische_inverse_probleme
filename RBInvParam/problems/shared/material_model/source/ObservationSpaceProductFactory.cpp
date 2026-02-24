#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/matrix_tools.h>

#include "ObservationSpaceProductFactory.hpp"

template class ObservationSpaceProductFactory<3, double>;

const std::unordered_map<ObservationSpaceProductType, FEProductType> obsToState {
    { ObservationSpaceProductType::STATE_L2,        FEProductType::L2 },
    { ObservationSpaceProductType::STATE_L2_0,      FEProductType::L2_0 },
    { ObservationSpaceProductType::STATE_H1_semi,   FEProductType::H1_semi },
    { ObservationSpaceProductType::STATE_H1_0_semi, FEProductType::H1_0_semi },
    { ObservationSpaceProductType::STATE_H1,        FEProductType::H1 },
    { ObservationSpaceProductType::STATE_H1_0,      FEProductType::H1_0 },
};


template <int dim, typename Number>
void ObservationSpaceProductFactory<dim, Number>::assemble_observation_space_product(
  const ObservationSpaceProductFactoryContext<dim, Number>& ctx,
  SparseMatrix<Number>& observation_space_product_matrix,
  SparsityPattern& observation_space_product_sp) const
{
  switch (ctx.obs_space_product_type)
  {
  case ObservationSpaceProductType::EUCLID:
    ObservationSpaceProductFactory::assemble_euclid_product(
        ctx,
        observation_space_product_matrix,
        observation_space_product_sp
    );
    break;
  case ObservationSpaceProductType::STATE_L2:
  case ObservationSpaceProductType::STATE_L2_0:
  case ObservationSpaceProductType::STATE_H1_semi:
  case ObservationSpaceProductType::STATE_H1_0_semi:
  case ObservationSpaceProductType::STATE_H1:
  case ObservationSpaceProductType::STATE_H1_0:
    ObservationSpaceProductFactory::assemble_product(
        ctx,
        observation_space_product_matrix,
        observation_space_product_sp
    );
    break;
  default:
    throw std::runtime_error("Unknown ObservationSpaceProductType.");
  }
};

template <int dim, typename Number>
void ObservationSpaceProductFactory<dim, Number>::assemble_euclid_product(
  const ObservationSpaceProductFactoryContext<dim, Number>& ctx,
  SparseMatrix<Number>& observation_space_product_matrix,
  SparsityPattern& observation_space_product_sp) const
{
  const unsigned int n = ctx.observation_space_dim;
  DynamicSparsityPattern dsp(n, n);
    for (types::global_dof_index i = 0; i < n; ++i)
      dsp.add(i, i);

    observation_space_product_sp.copy_from(dsp);
    observation_space_product_matrix.reinit(observation_space_product_sp);

    for (types::global_dof_index i = 0; i < n; ++i)
      observation_space_product_matrix.set(i, i, Number(1));
};

template <int dim, typename Number>
void ObservationSpaceProductFactory<dim, Number>::assemble_product(
  const ObservationSpaceProductFactoryContext<dim, Number>& ctx,
  SparseMatrix<Number>& observation_space_product_matrix,
  SparsityPattern& observation_space_product_sp) const
{
    AssertThrow(ctx.observation_space_dim == ctx.state_sparsity_pattern.n_rows() &&
                ctx.observation_space_dim == ctx.state_sparsity_pattern.n_cols(),
                ExcMessage("Dimension of the observation space does not coincide with dimension of the state space"));

    // FEProductType state_product_type = obsToState.find(ctx.obs_space_product_type)->second;
    // ProductFactoryContext<dim, Number> state_product_factory_ctx {
    //     state_product_type,
    //     ctx.fe,
    //     ctx.dof_handler,
    //     ctx.state_sparsity_pattern
    // };

    // observation_space_product_sp.copy_from(ctx.state_sparsity_pattern);
    // m_state_product_factory.assemble_product(
    //     state_product_factory_ctx,
    //     observation_space_product_matrix
    // );
};
  