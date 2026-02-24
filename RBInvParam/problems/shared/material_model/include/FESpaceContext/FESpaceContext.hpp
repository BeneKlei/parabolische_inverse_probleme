#pragma once

#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_pattern.h>

template <int dim, typename Number>
class FESpaceContext
{
public:
  FESpaceContext(const dealii::FiniteElement<dim>         &fe,
                 const dealii::DoFHandler<dim>            &dof_handler,
                 const dealii::Mapping<dim>               &mapping,
                 const dealii::Quadrature<dim>            &quadrature,
                 const dealii::SparsityPattern            &sparsity_pattern,
                 const dealii::AffineConstraints<Number>  &constraints)
    : m_fe(fe)
    , m_dof_handler(dof_handler)
    , m_mapping(mapping)
    , m_quadrature(quadrature)
    , m_sparsity_pattern(sparsity_pattern)
    , m_constraints(constraints)
  {}

  const dealii::FiniteElement<dim>        &fe() const { return m_fe; }
  const dealii::DoFHandler<dim>           &dof_handler() const { return m_dof_handler; }
  const dealii::Mapping<dim>              &mapping() const { return m_mapping; }
  const dealii::Quadrature<dim>           &quadrature() const { return m_quadrature; }
  const dealii::SparsityPattern           &sparsity_pattern() const { return m_sparsity_pattern; }
  const dealii::AffineConstraints<Number> &constraints() const { return m_constraints; }

  unsigned int n_dofs() const { return m_dof_handler.n_dofs(); }
  unsigned int n_components() const { return m_fe.n_components(); }

private:
  const dealii::FiniteElement<dim>         &m_fe;
  const dealii::DoFHandler<dim>            &m_dof_handler;
  const dealii::Mapping<dim>               &m_mapping;
  const dealii::Quadrature<dim>            &m_quadrature;
  const dealii::SparsityPattern            &m_sparsity_pattern;
  const dealii::AffineConstraints<Number>  &m_constraints;
};