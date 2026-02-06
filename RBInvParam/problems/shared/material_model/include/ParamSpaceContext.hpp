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
  ParamSpaceContext(const Triangulation<dim>                        &triangulation,
                    const FE_Q<dim>                                 &fe,
                    const DoFHandler<dim>                           &dof_handler,
                    const Mapping<dim>                              &mapping,
                    const Quadrature<dim>                           &quadrature,
                    //Utilities::MPI::RemotePointEvaluation<dim, dim> &param_rpe,
                    const AffineConstraints<Number>                 &constraints,
                    const std::vector<types::global_dof_index>      &free_dofs,
                    const std::vector<unsigned int>                  grid_resolution,
                    const Point<dim>                                 p1,
                    const Point<dim>                                 p2)
    : m_triangulation(triangulation)
    , m_fe(fe)
    , m_dof_handler(dof_handler)
    , m_mapping(mapping)
    , m_quadrature(quadrature)
    //, m_rpe(param_rpe)
    , m_constraints(constraints)
    , m_free_dofs(free_dofs)
    , m_grid_resolution(grid_resolution)
    , m_p1(p1)
    , m_p2(p2)
  {}

  const Triangulation<dim> & triangulation() const { return m_triangulation; }
  const FE_Q<dim>          & fe()            const { return m_fe; }
  const DoFHandler<dim>    & dof_handler()   const { return m_dof_handler; }
  const Mapping<dim>       & mapping()       const { return m_mapping; }
  const Quadrature<dim>    & quadrature()    const { return m_quadrature; }

  const AffineConstraints<Number> & constraints() const { return m_constraints; }
  const std::vector<types::global_dof_index> & free_dofs() const { return m_free_dofs; }


  void evaluate_values(
    const Vector<Number>          &param,
    const std::vector<Point<dim>> &points,
    std::vector<Number>           &values,
    bool                           linear_part = false
  ) const;

  Number evaluate_value(
    const Vector<Number>  &param,
    const Point<dim>      &point,
    bool                   linear_part = false
  ) const;

  void project_to_free_param(
    const Vector<Number> &full_param,
    Vector<Number>       &free_param
  ) const;

  // void reconstruct_full_param(
  //   const Vector<Number>          &param,
  //   Vector<Number>                &param_full
  // ) const;


private:
  // Geometry / discretization
  const Triangulation<dim>                       &m_triangulation;
  const Mapping<dim>                             &m_mapping;
  const FE_Q<dim>                                &m_fe;
  const DoFHandler<dim>                          &m_dof_handler;
  const Quadrature<dim>                          &m_quadrature;

  // Constraints and parameter indexing
  const AffineConstraints<Number>                &m_constraints;
  const std::vector<types::global_dof_index>     &m_free_dofs;

  // Parameter grid metadata (owned by this context)
  const std::vector<unsigned int>                 m_grid_resolution;
  const Point<dim>                                m_p1;
  const Point<dim>                                m_p2;
};