#pragma once

#include <vector>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/tria.h>

#include "FESpaceContext.hpp"

template <int dim, typename Number>
class StateSpaceContext : public FESpaceContext<dim, Number, dealii::FESystem<dim>>
{
public:
  using Base = FESpaceContext<dim, Number, dealii::FESystem<dim>>;

  StateSpaceContext(const dealii::Triangulation<dim>        &triangulation,
                    const dealii::FESystem<dim>             &fe,
                    const dealii::DoFHandler<dim>           &dof_handler,
                    const dealii::Mapping<dim>              &mapping,
                    const dealii::Quadrature<dim>           &quadrature,
                    const dealii::SparsityPattern           &sparsity_pattern,
                    const dealii::AffineConstraints<Number> &boundary_constraints,
                    std::vector<unsigned int>                grid_resolution,
                    const dealii::Point<dim>                 p1,
                    const dealii::Point<dim>                 p2)
    : Base(fe, dof_handler, mapping, quadrature, sparsity_pattern, boundary_constraints)
    , m_triangulation(triangulation)
    , m_grid_resolution(std::move(grid_resolution))
    , m_p1(p1)
    , m_p2(p2)
  {}

  // Intent-revealing alias:
  const dealii::AffineConstraints<Number> &boundary_constraints() const
  {
    return this->constraints();
  }

  // (Optionally) hide the generic name to force call sites to use the explicit one:
  // using Base::constraints;   // keep it visible
  // OR:
  // private: using Base::constraints;  // hide it

  const dealii::Triangulation<dim> &triangulation() const { return m_triangulation; }

  void pre_compute();

  const std::vector<dealii::Point<dim>> &quad_points_flat() const
  {
    Assert(m_pre_computed,
           dealii::ExcMessage("StateSpaceContext not precomputed. Call pre_compute()."));
    return m_quad_points_flat;
  }

private:
  void pre_compute_quad_points_flat();

  const dealii::Triangulation<dim> &m_triangulation;

  std::vector<unsigned int> m_grid_resolution;
  dealii::Point<dim>        m_p1;
  dealii::Point<dim>        m_p2;

  std::vector<dealii::Point<dim>> m_quad_points_flat;
  bool m_pre_computed = false;
};