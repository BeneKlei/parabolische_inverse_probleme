#pragma once

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/dofs/dof_handler.h>

using namespace dealii;

enum class StateProductType {
    L2,
    L2_0,
    H1_semi,
    H1_0_semi,
    H1,
    H1_0,
    Mass,
    BoundaryMass
};

template <int dim, typename Number>
struct StateProductFactoryContext {
  const StateProductType             &state_product_type;
  const FiniteElement<dim>           &fe;
  const DoFHandler<dim>              &dof_handler;
  const SparsityPattern              &sparsity_pattern;
};

template <int dim, typename Number>
class StateProductFactory
{
public:
    void assemble_state_product(
        const StateProductFactoryContext<dim, Number>& ctx,
        SparseMatrix<Number>& state_product_matrix
    ) const;

    void assemble_l2_product(
        const StateProductFactoryContext<dim, Number>& ctx,
        SparseMatrix<Number>& state_product_matrix
    ) const;

    void assemble_l2_0_product(
        const StateProductFactoryContext<dim, Number>& ctx,
        SparseMatrix<Number>& state_product_matrix
    ) const;

    void assemble_h1_semi_product(
        const StateProductFactoryContext<dim, Number>& ctx,
        SparseMatrix<Number>& state_product_matrix
    ) const;

    void assemble_h1_0_semi_product(
        const StateProductFactoryContext<dim, Number>& ctx,
        SparseMatrix<Number>& state_product_matrix
    ) const;

    void assemble_h1_product(
        const StateProductFactoryContext<dim, Number>& ctx,
        SparseMatrix<Number>& state_product_matrix
    ) const;

    void assemble_h1_0_product(
        const StateProductFactoryContext<dim, Number>& ctx,
        SparseMatrix<Number>& state_product_matrix
    ) const;

    void assemble_mass_product(
        const StateProductFactoryContext<dim, Number>& ctx,
        SparseMatrix<Number>& state_product_matrix
    ) const;

    // void assemble_boundary_mass_product(
    //     const StateProductFactoryContext<dim, Number>& ctx,
    //     SparseMatrix<Number>& state_product_matrix
    // ) const;

    template <typename Integrand>
    void _assemble_product(
        const StateProductFactoryContext<dim, Number>& ctx,
        SparseMatrix<Number>& state_product_matrix,
        Integrand integrand,
        const AffineConstraints<Number>& constraints
    ) const;

};