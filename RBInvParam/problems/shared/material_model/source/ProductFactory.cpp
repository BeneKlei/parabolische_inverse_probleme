#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/vector_operation.h>

#include <deal.II/fe/fe_values.h>

#include <stdexcept>
#include <vector>

#include "ProductFactory.hpp"

template <int dim, typename Number>
void ProductFactory<dim, Number>::assemble_l2(
    const Ctx& ctx, dealii::SparseMatrix<Number>& M) const
{
  auto integrand =
    [](unsigned int i, unsigned int j, unsigned int q,
       const dealii::FEValues<dim>& fe_values)
    {
      return fe_values.shape_value(i, q) * fe_values.shape_value(j, q);
    };

  dealii::AffineConstraints<Number> empty;
  empty.clear();
  empty.close();

  assemble_cell_product(ctx, M, integrand, empty);
}


template <int dim, typename Number>
void ProductFactory<dim, Number>::assemble_l2_0(
    const Ctx& ctx, dealii::SparseMatrix<Number>& M) const
{
  auto integrand =
    [](unsigned int i, unsigned int j, unsigned int q,
       const dealii::FEValues<dim>& fe_values)
    {
      return fe_values.shape_value(i, q) * fe_values.shape_value(j, q);
    };

  dealii::AffineConstraints<Number> zero;
  dealii::Functions::ZeroFunction<dim> zero_fun(ctx.space.n_components());

  for (const auto id : ctx.space.dof_handler().get_triangulation().get_boundary_ids())
    dealii::VectorTools::interpolate_boundary_values(
        ctx.space.dof_handler(), id, zero_fun, zero);

  zero.close();

  assemble_cell_product(ctx, M, integrand, zero);
}


template <int dim, typename Number>
void ProductFactory<dim, Number>::assemble_h1_semi(
    const Ctx& ctx, dealii::SparseMatrix<Number>& M) const
{
  auto integrand =
    [](unsigned int i, unsigned int j, unsigned int q,
       const dealii::FEValues<dim>& fe_values)
    {
      return fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q);
    };

  dealii::AffineConstraints<Number> empty;
  empty.clear();
  empty.close();

  assemble_cell_product(ctx, M, integrand, empty);
}

template <int dim, typename Number>
void ProductFactory<dim, Number>::assemble_h1_0_semi(
    const Ctx& ctx, dealii::SparseMatrix<Number>& M) const
{
  auto integrand =
    [](unsigned int i, unsigned int j, unsigned int q,
       const dealii::FEValues<dim>& fe_values)
    {
      return fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q);
    };

  dealii::AffineConstraints<Number> zero;
  dealii::Functions::ZeroFunction<dim> zero_fun(ctx.space.n_components());

  for (const auto id : ctx.space.dof_handler().get_triangulation().get_boundary_ids())
    dealii::VectorTools::interpolate_boundary_values(
        ctx.space.dof_handler(), id, zero_fun, zero);

  zero.close();

  assemble_cell_product(ctx, M, integrand, zero);
}


template <int dim, typename Number>
void ProductFactory<dim, Number>::assemble_h1(
    const Ctx& ctx, dealii::SparseMatrix<Number>& M) const
{
  auto integrand =
    [](unsigned int i, unsigned int j, unsigned int q,
       const dealii::FEValues<dim>& fe_values)
    {
      return fe_values.shape_value(i, q) * fe_values.shape_value(j, q)
           + fe_values.shape_grad(i, q)  * fe_values.shape_grad(j, q);
    };

  dealii::AffineConstraints<Number> empty;
  empty.clear();
  empty.close();

  assemble_cell_product(ctx, M, integrand, empty);
}

template <int dim, typename Number>
void ProductFactory<dim, Number>::assemble_h1_0(
    const Ctx& ctx, dealii::SparseMatrix<Number>& M) const
{
  // Define H1_0 as full H1 inner product with zero trace enforced.
  auto integrand =
    [](unsigned int i, unsigned int j, unsigned int q,
       const dealii::FEValues<dim>& fe_values)
    {
      return fe_values.shape_value(i, q) * fe_values.shape_value(j, q)
           + fe_values.shape_grad(i, q)  * fe_values.shape_grad(j, q);
    };

  dealii::AffineConstraints<Number> zero;
  dealii::Functions::ZeroFunction<dim> zero_fun(ctx.space.n_components());

  for (const auto id : ctx.space.dof_handler().get_triangulation().get_boundary_ids())
    dealii::VectorTools::interpolate_boundary_values(
        ctx.space.dof_handler(), id, zero_fun, zero);

  zero.close();

  assemble_cell_product(ctx, M, integrand, zero);
}

template <int dim, typename Number>
void ProductFactory<dim, Number>::assemble_mass(
    const Ctx& ctx, dealii::SparseMatrix<Number>& M) const
{
  assemble_l2(ctx, M);
}

template <int dim, typename Number>
void ProductFactory<dim, Number>::assemble_boundary_mass(
    const Ctx& ctx, dealii::SparseMatrix<Number>& M) const
{
  const auto& fe  = ctx.space.fe();
  const auto& dof = ctx.space.dof_handler();

  // IMPORTANT: match your original behavior: EMPTY constraints.
  dealii::AffineConstraints<Number> empty;
  empty.clear();
  empty.close();

  dealii::QGaussLobatto<dim - 1> face_quad(2);

  dealii::FEFaceValues<dim> fe_face_values(
      ctx.space.mapping(),
      fe,
      face_quad,
      dealii::update_values | dealii::update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q = face_quad.size();

  dealii::FullMatrix<Number> cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

  M.reinit(ctx.space.sparsity_pattern());
  M = 0;

  for (auto cell = dof.begin_active(); cell != dof.end(); ++cell)
  {
    cell->get_dof_indices(local_dof_indices);

    for (unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
    {
      if (!cell->face(face)->at_boundary())
        continue;

      fe_face_values.reinit(cell, face);
      cell_matrix = 0;

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const unsigned int comp_i = fe.system_to_component_index(i).first;

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          const unsigned int comp_j = fe.system_to_component_index(j).first;
          if (comp_i != comp_j) continue;

          for (unsigned int q = 0; q < n_q; ++q)
            cell_matrix(i, j) += fe_face_values.shape_value(i, q)
                               * fe_face_values.shape_value(j, q)
                               * fe_face_values.JxW(q);
        }
      }

      empty.distribute_local_to_global(cell_matrix, local_dof_indices, M);
    }
  }

  empty.condense(M);
}


template <int dim, typename Number>
template <typename Integrand>
void ProductFactory<dim, Number>::assemble_cell_product(
    const Ctx& ctx,
    dealii::SparseMatrix<Number>& M,
    Integrand integrand,
    const dealii::AffineConstraints<Number>& constraints) const
{
  const auto& fe = ctx.space.fe();
  const auto& dof = ctx.space.dof_handler();

  // Product quadrature (keep your previous choice; not necessarily ctx.space.quadrature())
  dealii::QGaussLobatto<dim> quad(2);

  dealii::FEValues<dim> fe_values(
      ctx.space.mapping(),
      fe,
      quad,
      dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q = quad.size();

  dealii::FullMatrix<Number> cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

  M.reinit(ctx.space.sparsity_pattern());
  M = 0;

  for (auto cell = dof.begin_active(); cell != dof.end(); ++cell)
  {
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell->get_dof_indices(local_dof_indices);

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      const unsigned int comp_i = fe.system_to_component_index(i).first;

      for (unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        const unsigned int comp_j = fe.system_to_component_index(j).first;
        if (comp_i != comp_j) continue;

        for (unsigned int q = 0; q < n_q; ++q)
          cell_matrix(i, j) += integrand(i, j, q, fe_values) * fe_values.JxW(q);
      }
    }

    constraints.distribute_local_to_global(cell_matrix, local_dof_indices, M);
  }

  constraints.condense(M);
}
