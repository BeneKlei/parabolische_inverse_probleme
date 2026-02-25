#pragma once

#include <vector>

#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/vector.h>

#include "FESpaceContext/FESpaceContext.hpp"

template <int dim, typename Number>
class ParamSpaceContext
  : public FESpaceContext<dim, Number>
{
public:
  using Base = FESpaceContext<dim, Number>;

  ParamSpaceContext(
      const dealii::Triangulation<dim>                  &triangulation,
      const dealii::FE_Q<dim>                           &fe,
      const dealii::DoFHandler<dim>                     &dof_handler,
      const dealii::Mapping<dim>                        &mapping,
      const dealii::Quadrature<dim>                     &quadrature,
      const dealii::SparsityPattern                     &sparsity_pattern,
      const dealii::AffineConstraints<Number>           &constraints,
      std::vector<unsigned int>                          grid_resolution,
      const dealii::Point<dim>                           p1,
      const dealii::Point<dim>                           p2)
    : Base(fe, dof_handler, mapping, quadrature, sparsity_pattern, constraints)
    , m_triangulation(triangulation)
    , m_grid_resolution(std::move(grid_resolution))
    , m_p1(p1)
    , m_p2(p2)
  {}

  const dealii::Triangulation<dim> &triangulation() const { return m_triangulation; }
  const std::vector<dealii::types::global_dof_index> &free_dofs() const { return m_free_dofs; }

  void compute_free_dofs();

  void build_index_maps();

  // Convert between representations
  void free_to_grid(const dealii::Vector<Number> &param_free,
                    dealii::Vector<Number>       &param_grid,
                    bool                          linear_part) const;

  void grid_to_free(const dealii::Vector<Number> &param_grid,
                    dealii::Vector<Number>       &param_free) const;

  // -----------------------------------------------

  void evaluate_values(
    const dealii::Vector<Number>          &param,
    const std::vector<dealii::Point<dim>> &points,
    std::vector<Number>                   &values,
    bool linear_part = false
  ) const;

  Number evaluate_value(
    const dealii::Vector<Number> &param,
    const dealii::Point<dim>     &point,
    bool linear_part = false
  ) const;

  void extract_principal_submatrix(
      const dealii::SparseMatrix<Number>& full_matrix,
      dealii::SparseMatrix<Number>& reduced_matrix,
      dealii::SparsityPattern& reduced_param_sp
  ) const;

  void project_to_free_param(
    const dealii::Vector<Number> &full_param,
    dealii::Vector<Number>       &free_param
  ) const;

private:
  const dealii::Triangulation<dim>                   &m_triangulation;
  std::vector<dealii::types::global_dof_index>        m_free_dofs;
  std::vector<int>                                    m_global_to_free;
  std::vector<dealii::types::global_dof_index>        m_grid_slot_to_global;

  std::vector<unsigned int> m_grid_resolution;
  dealii::Point<dim>        m_p1;
  dealii::Point<dim>        m_p2;


  unsigned int ny() const { return m_grid_resolution[1]; }
  unsigned int nz() const { return m_grid_resolution[2]; }
  unsigned int stride() const { return ny() + 1; }
  unsigned int n_grid_slots() const { return (ny() + 1) * (nz() + 1); }

  unsigned int grid_slot_from_point(const dealii::Point<dim> &point) const;
};