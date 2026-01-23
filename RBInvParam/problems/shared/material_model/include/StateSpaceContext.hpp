#pragma once

#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/sparsity_pattern.h>

using namespace dealii;

template <int dim>
class StateSpaceContext
{
public:
  StateSpaceContext(const Triangulation<dim> &triangulation,
                    const FESystem<dim>      &fe,
                    const DoFHandler<dim>    &dof_handler,
                    const SparsityPattern    &state_sp)
    : m_triangulation(triangulation)
    , m_fe(fe)
    , m_dof_handler(dof_handler)
    , m_state_sp(state_sp)
  {}

  const Triangulation<dim> & triangulation() const { return m_triangulation; }
  const FESystem<dim>      & fe()            const { return m_fe; }
  const DoFHandler<dim>    & dof_handler()   const { return m_dof_handler; }
  const SparsityPattern    & state_sp()      const { return m_state_sp; }

  unsigned int n_dofs() const { return m_dof_handler.n_dofs(); }

private:
  const Triangulation<dim> & m_triangulation;
  const FESystem<dim>      & m_fe;
  const DoFHandler<dim>    & m_dof_handler;
  const SparsityPattern    & m_state_sp;
};
