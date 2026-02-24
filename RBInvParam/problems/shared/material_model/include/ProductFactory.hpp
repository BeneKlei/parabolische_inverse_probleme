#pragma once

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/numerics/vector_tools.h>

#include "FESpaceContext/StateSpaceContext.hpp"   // your context

enum class FEProductType {
  L2, L2_0,
  H1_semi, H1_0_semi,
  H1, H1_0,
  Mass,
  BoundaryMass
};

template <int dim, typename Number>
struct ProductFactoryContext
{
  FEProductType type;
  const FESpaceContext<dim, Number>& space;
};

template <int dim, typename Number>
class ProductFactory
{
public:
  using Ctx = ProductFactoryContext<dim, Number>;

  void assemble_product(const Ctx& ctx,
                        dealii::SparseMatrix<Number>& M) const
  {
    switch (ctx.type)
    {
      case FEProductType::L2:          assemble_l2(ctx, M); break;
      case FEProductType::L2_0:        assemble_l2_0(ctx, M); break;
      case FEProductType::H1_semi:     assemble_h1_semi(ctx, M); break;
      case FEProductType::H1_0_semi:   assemble_h1_0_semi(ctx, M); break;
      case FEProductType::H1:          assemble_h1(ctx, M); break;
      case FEProductType::H1_0:        assemble_h1_0(ctx, M); break;
      case FEProductType::Mass:        assemble_mass(ctx, M); break;
      case FEProductType::BoundaryMass:assemble_boundary_mass(ctx, M); break;
      default: throw std::runtime_error("Unknown product type.");
    }
  }

private:
  void assemble_l2(const Ctx& ctx, dealii::SparseMatrix<Number>& M) const;
  void assemble_l2_0(const Ctx& ctx, dealii::SparseMatrix<Number>& M) const;
  void assemble_h1_semi(const Ctx& ctx, dealii::SparseMatrix<Number>& M) const;
  void assemble_h1_0_semi(const Ctx& ctx, dealii::SparseMatrix<Number>& M) const;
  void assemble_h1(const Ctx& ctx, dealii::SparseMatrix<Number>& M) const;
  void assemble_h1_0(const Ctx& ctx, dealii::SparseMatrix<Number>& M) const;
  void assemble_mass(const Ctx& ctx, dealii::SparseMatrix<Number>& M) const;
  void assemble_boundary_mass(const Ctx& ctx, dealii::SparseMatrix<Number>& M) const;

  template <typename Integrand>
  void assemble_cell_product(const Ctx& ctx,
                             dealii::SparseMatrix<Number>& M,
                             Integrand integrand,
                             const dealii::AffineConstraints<Number>& constraints) const;
};

template class ProductFactory<3, double>;