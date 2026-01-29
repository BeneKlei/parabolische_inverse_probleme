#pragma once

#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/sparsity_pattern.h>

using namespace dealii;

template <int dim, typename Number>
class StateSpaceContext
{
public:
  StateSpaceContext(const Triangulation<dim>        &triangulation,
                    const FESystem<dim>             &fe,
                    const DoFHandler<dim>           &dof_handler,
                    const Quadrature<dim>           &quadrature,
                    const SparsityPattern           &state_sp,
                    const AffineConstraints<Number> &BC_constraints)
    : m_state_triangulation(triangulation)
    , m_fe(fe)
    , m_dof_handler(dof_handler)
    , m_quadrature(quadrature)
    , m_state_sp(state_sp)
    , m_BC_constraints(BC_constraints)
  {}

  const Triangulation<dim>           & triangulation()  const { return m_state_triangulation; }
  const FESystem<dim>                & fe()             const { return m_fe; }
  const DoFHandler<dim>              & dof_handler()    const { return m_dof_handler; }
  const Quadrature<dim>              & quadrature()     const { return m_quadrature; }
  const SparsityPattern              & state_sp()       const { return m_state_sp; }
  const AffineConstraints<Number>    & BC_constraints() const { return m_BC_constraints; }

  const std::vector<Point<dim>>      & quad_points_flat() const {     
    Assert(m_pre_computed,
       ExcMessage("StateSpaceContext not precomputed. Call precompute()."));
    return m_quad_points_flat; 
  }
  // const std::vector<unsigned int>    & cell_offsets() const { 
  //   Assert(m_pre_computed,
  //      ExcMessage("StateSpaceContext not precomputed. Call precompute()."));
  //   return m_cell_offsets; 
  // }

  unsigned int n_dofs() const { return m_dof_handler.n_dofs(); }
  void pre_compute();

private:
  void pre_compute_quad_points_flat();

  const Triangulation<dim>        & m_state_triangulation;
  const FESystem<dim>             & m_fe;
  const DoFHandler<dim>           & m_dof_handler;
  const Quadrature<dim>           & m_quadrature;
  const SparsityPattern           & m_state_sp;
  const AffineConstraints<Number> & m_BC_constraints;

  std::vector<Point<dim>>           m_quad_points_flat;
  //std::vector<unsigned int>         m_cell_offsets;

  bool m_pre_computed = false;
};
