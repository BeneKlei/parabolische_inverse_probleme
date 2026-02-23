#pragma once

#include <vector>

#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/vector.h>

#include "FESpaceContext/FESpaceContext.hpp"

template <int dim, typename Number>
class ParamSpaceContext
  : public FESpaceContext<dim, Number, dealii::FE_Q<dim>>
{
public:
  using Base = FESpaceContext<dim, Number, dealii::FE_Q<dim>>;

  ParamSpaceContext(
      const dealii::Triangulation<dim>                  &triangulation,
      const dealii::FE_Q<dim>                           &fe,
      const dealii::DoFHandler<dim>                     &dof_handler,
      const dealii::Mapping<dim>                        &mapping,
      const dealii::Quadrature<dim>                     &quadrature,
      const dealii::SparsityPattern                     &sparsity_pattern,
      const dealii::AffineConstraints<Number>           &constraints,
      const std::vector<dealii::types::global_dof_index> &free_dofs,
      std::vector<unsigned int>                          grid_resolution,
      const dealii::Point<dim>                           p1,
      const dealii::Point<dim>                           p2)
    : Base(fe, dof_handler, mapping, quadrature, sparsity_pattern, constraints)
    , m_triangulation(triangulation)
    , m_free_dofs(free_dofs)
    , m_grid_resolution(std::move(grid_resolution))
    , m_p1(p1)
    , m_p2(p2)
  {}

  const dealii::Triangulation<dim> &triangulation() const { return m_triangulation; }
  const std::vector<dealii::types::global_dof_index> &free_dofs() const { return m_free_dofs; }

  void evaluate_values(const dealii::Vector<Number>          &param,
                       const std::vector<dealii::Point<dim>> &points,
                       std::vector<Number>                   &values,
                       bool linear_part = false) const;

  Number evaluate_value(const dealii::Vector<Number> &param,
                        const dealii::Point<dim>     &point,
                        bool linear_part = false) const;

  void project_to_free_param(const dealii::Vector<Number> &full_param,
                             dealii::Vector<Number>       &free_param) const;

private:
  const dealii::Triangulation<dim>                  &m_triangulation;
  const std::vector<dealii::types::global_dof_index> &m_free_dofs;

  std::vector<unsigned int> m_grid_resolution;
  dealii::Point<dim>        m_p1;
  dealii::Point<dim>        m_p2;
};