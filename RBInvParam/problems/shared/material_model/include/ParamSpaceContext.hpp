#pragma once

#include <memory>
#include <vector>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/affine_constraints.h>

using namespace dealii;


template <int dim, typename Number>
class ParamSpaceContext
{
public:
  ParamSpaceContext(const Triangulation<dim> &triangulation,
                    const FE_Q<dim>          &fe,
                    const DoFHandler<dim>    &dof_handler,
                    const MappingQ1<dim>     &mapping,
                    FEPointEvaluation<1, dim>      &evaluator,
                    const std::unique_ptr<GridTools::Cache<dim>> &grid_cache,
                    const AffineConstraints<Number> &constraints,
                    const std::vector<types::global_dof_index>    &free_dofs)
    : m_triangulation(triangulation)
    , m_fe(fe)
    , m_dof_handler(dof_handler)
    , m_mapping(mapping)
    , m_evaluator(evaluator)
    , m_grid_cache(grid_cache)
    , m_constraints(constraints)
    , m_free_dofs(free_dofs)
  {}

  const Triangulation<dim> & triangulation() const { return m_triangulation; }
  const FE_Q<dim>          & fe()            const { return m_fe; }
  const DoFHandler<dim>    & dof_handler()   const { return m_dof_handler; }
  const MappingQ1<dim>     & mapping()       const { return m_mapping; }

  const AffineConstraints<Number> & constraints() const { return m_constraints; }
  const std::vector<types::global_dof_index> & free_dofs() const { return m_free_dofs; }

  bool grid_cache_initialized() const
  {
    return (m_grid_cache.get() != nullptr);
  }

  void evaluate_values(
    const Vector<Number>          &param_full,
    const std::vector<Point<dim>> &points,
    std::vector<Number>           &values
  ) const;

  void evaluate_values_from_reduced(
    const std::vector<Number>     &param_reduced,
    Vector<Number>                &full_buffer,
    const std::vector<Point<dim>> &points,
    std::vector<Number>           &values
  ) const;

private:
  const Triangulation<dim> & m_triangulation;
  const FE_Q<dim>          & m_fe;
  const DoFHandler<dim>    & m_dof_handler;

  const MappingQ1<dim>     & m_mapping;

  FEPointEvaluation<1, dim> & m_evaluator;
  const std::unique_ptr<GridTools::Cache<dim>> & m_grid_cache;

  const AffineConstraints<Number> & m_constraints;
  const std::vector<types::global_dof_index>    & m_free_dofs;
};