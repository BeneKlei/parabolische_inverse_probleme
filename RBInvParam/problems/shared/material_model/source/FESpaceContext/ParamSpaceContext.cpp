#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <vector>

#include <deal.II/base/point.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h> // or AffineConstraints

#include <deal.II/fe/mapping.h>

#include <deal.II/numerics/fe_field_function.h>

#include "FESpaceContext/ParamSpaceContext.hpp"


template <int dim, typename Number>
void ParamSpaceContext<dim, Number>::compute_free_dofs()
{
  const auto n_dofs = this->dof_handler().n_dofs();

  m_free_dofs.clear();
  m_free_dofs.reserve(n_dofs - this->constraints().n_constraints());

  for (dealii::types::global_dof_index i = 0; i < n_dofs; ++i)
    if (!this->constraints().is_constrained(i))
      m_free_dofs.push_back(i);
}

template <int dim, typename Number>
unsigned int ParamSpaceContext<dim, Number>::grid_slot_from_point(
  const dealii::Point<dim> &point) const
{
  const unsigned int Ny = ny();
  const unsigned int Nz = nz();
  const double y_min = m_p1[1], y_max = m_p2[1];
  const double z_min = m_p1[2], z_max = m_p2[2];

  auto clamp = [](int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
  };

  const int iy = clamp(
    static_cast<int>(std::lround((point[1] - y_min) / (y_max - y_min) * Ny)),
    0, static_cast<int>(Ny));

  const int iz = clamp(
    static_cast<int>(std::lround((point[2] - z_min) / (z_max - z_min) * Nz)),
    0, static_cast<int>(Nz));

  return static_cast<unsigned int>(iy) + static_cast<unsigned int>(iz) * stride();
}

template <int dim, typename Number>
void ParamSpaceContext<dim, Number>::build_index_maps()
{
  // 1) global_to_free
  {
    const auto n_dofs = this->dof_handler().n_dofs();
    m_global_to_free.assign(n_dofs, -1);
    for (unsigned int i_free = 0; i_free < m_free_dofs.size(); ++i_free)
      m_global_to_free[m_free_dofs[i_free]] = static_cast<int>(i_free);
  }

  // 2) grid_slot_to_global (dense grid on plane x==p1[0])
  m_grid_slot_to_global.assign(n_grid_slots(), dealii::numbers::invalid_dof_index);

  // Map dofs to (support) points.
  // Works well for nodal elements (e.g., FE_Q). For non-nodal elements,
  // you need a different identification strategy.
  std::map<dealii::types::global_dof_index, dealii::Point<dim>> support_points;
  dealii::DoFTools::map_dofs_to_support_points(this->mapping(),
                                               this->dof_handler(),
                                               support_points);

  const double tol_x = 1e-12;

  for (const auto &kv : support_points)
  {
    const auto dof = kv.first;
    const auto &pt = kv.second;

    // only use DoFs whose support point lies on x == p1[0] plane
    if (std::abs(pt[0] - m_p1[0]) > tol_x)
      continue;

    const unsigned int slot = grid_slot_from_point(pt);

    // If two DoFs map to same slot, that indicates your FE/space is not
    // "one DoF per grid node" on that plane (or tolerances too loose).
    // Prefer throwing to avoid silent corruption.
    if (m_grid_slot_to_global[slot] != dealii::numbers::invalid_dof_index &&
        m_grid_slot_to_global[slot] != dof)
    {
      AssertThrow(false, dealii::ExcMessage(
        "Multiple DoFs mapped to the same (iy,iz) grid slot. "
        "Geometric grid interpretation is not compatible with this FE/DoF layout."));
    }

    m_grid_slot_to_global[slot] = dof;
  }

  // Optional sanity: ensure all slots are filled
  for (unsigned int slot = 0; slot < n_grid_slots(); ++slot)
  {
    AssertThrow(m_grid_slot_to_global[slot] != dealii::numbers::invalid_dof_index,
      dealii::ExcMessage("Some grid slots on x=p1[0] plane have no corresponding DoF. "
                         "Check mesh/FE/support point mapping and grid resolution."));
  }
}

template <int dim, typename Number>
void ParamSpaceContext<dim, Number>::free_to_grid(
  const dealii::Vector<Number> &param_free,
  dealii::Vector<Number>       &param_grid,
  bool                          linear_part) const
{
  AssertDimension(param_free.size(), m_free_dofs.size());
  AssertThrow(m_global_to_free.size() == this->dof_handler().n_dofs(),
              dealii::ExcMessage("Index maps not built. Call build_index_maps()."));
  AssertThrow(m_grid_slot_to_global.size() == n_grid_slots(),
              dealii::ExcMessage("Index maps not built. Call build_index_maps()."));

  param_grid.reinit(n_grid_slots());
  param_grid = Number(0);

  // Fill every geometric slot from corresponding free dof (or 0 if constrained)
  for (unsigned int slot = 0; slot < n_grid_slots(); ++slot)
  {
    const auto global = m_grid_slot_to_global[slot];
    const int free_i  = m_global_to_free[global];

    if (free_i >= 0)
      param_grid[slot] = param_free[static_cast<unsigned int>(free_i)];
    else
      param_grid[slot] = linear_part ? Number(0) : Number(1); // your policy for constrained slots
  }
}

template <int dim, typename Number>
void ParamSpaceContext<dim, Number>::grid_to_free(
  const dealii::Vector<Number> &param_grid,
  dealii::Vector<Number>       &param_free) const
{
  AssertDimension(param_grid.size(), n_grid_slots());
  AssertThrow(m_global_to_free.size() == this->dof_handler().n_dofs(),
              dealii::ExcMessage("Index maps not built. Call build_index_maps()."));
  AssertThrow(m_grid_slot_to_global.size() == n_grid_slots(),
              dealii::ExcMessage("Index maps not built. Call build_index_maps()."));

  param_free.reinit(m_free_dofs.size());
  param_free = Number(0);

  // Pull only free entries from geometric grid
  for (unsigned int slot = 0; slot < n_grid_slots(); ++slot)
  {
    const auto global = m_grid_slot_to_global[slot];
    const int free_i  = m_global_to_free[global];

    if (free_i >= 0)
      param_free[static_cast<unsigned int>(free_i)] = param_grid[slot];
  }
}

template <int dim, typename Number>
Number ParamSpaceContext<dim, Number>::evaluate_value(
  const dealii::Vector<Number> &param_grid,
  const dealii::Point<dim>     &point,
  bool                          linear_part) const
{
  AssertDimension(param_grid.size(), n_grid_slots());

  // Preserve your extrusion/affine boundary logic
  const double tol_x = 1e-12;
  if (std::abs(point[0] - m_p1[0]) > tol_x)
    return linear_part ? Number(0.0) : Number(1.0);

  const unsigned int slot = grid_slot_from_point(point);
  return param_grid[slot];
}

template <int dim, typename Number>
void ParamSpaceContext<dim, Number>::evaluate_values(
  const dealii::Vector<Number>          &param_grid,
  const std::vector<dealii::Point<dim>> &points,
  std::vector<Number>                   &values,
  bool                                   linear_part) const
{
  AssertDimension(param_grid.size(), n_grid_slots());
  values.resize(points.size());
  for (std::size_t i = 0; i < points.size(); ++i)
    values[i] = evaluate_value(param_grid, points[i], linear_part);
}

template <int dim, typename Number>
void ParamSpaceContext<dim, Number>::project_to_free_param(
  const dealii::Vector<Number> &full_param,
  dealii::Vector<Number>       &free_param) const
{
  AssertDimension(full_param.size(), this->dof_handler().n_dofs());
  free_param.reinit(m_free_dofs.size());

  for (unsigned int i = 0; i < m_free_dofs.size(); ++i)
    free_param[i] = full_param[m_free_dofs[i]];
}

template <int dim, typename Number>
void ParamSpaceContext<dim, Number>::extract_principal_submatrix(
    const dealii::SparseMatrix<Number>& full_matrix,
    dealii::SparseMatrix<Number>& reduced_matrix,
    dealii::SparsityPattern& reduced_param_sp
) const
{
  AssertThrow(!m_free_dofs.empty() || this->dof_handler().n_dofs() == this->constraints().n_constraints(),
            dealii::ExcMessage("m_free_dofs is empty; did you call compute_free_dofs()?"));

  AssertDimension(full_matrix.m(), this->dof_handler().n_dofs());
  AssertDimension(full_matrix.n(), this->dof_handler().n_dofs());

  const unsigned int n_reduced = m_free_dofs.size();

  // Map: global DoF index -> reduced DoF index
  std::vector<int> global_to_reduced(full_matrix.m(), -1);
  for (unsigned int reduced_index = 0; reduced_index < n_reduced; ++reduced_index)
  {
    const auto global_index = m_free_dofs[reduced_index];
    global_to_reduced[global_index] = static_cast<int>(reduced_index);
  }

  // Build sparsity pattern of reduced matrix
  dealii::DynamicSparsityPattern reduced_dsp(n_reduced, n_reduced);

  for (unsigned int reduced_row = 0; reduced_row < n_reduced; ++reduced_row)
  {
    const auto global_row = m_free_dofs[reduced_row];

    for (auto entry = full_matrix.begin(global_row);
         entry != full_matrix.end(global_row); ++entry)
    {
      const auto global_col = entry->column();
      const int reduced_col = global_to_reduced[global_col];

      if (reduced_col >= 0)
        reduced_dsp.add(reduced_row,
                        static_cast<unsigned int>(reduced_col));
    }
  }

  reduced_param_sp.copy_from(reduced_dsp);

  reduced_matrix.reinit(reduced_param_sp);
  reduced_matrix = 0.0;

  // Copy numerical values
  for (unsigned int reduced_row = 0; reduced_row < n_reduced; ++reduced_row)
  {
    const auto global_row = m_free_dofs[reduced_row];

    for (auto entry = full_matrix.begin(global_row);
         entry != full_matrix.end(global_row); ++entry)
    {
      const auto global_col = entry->column();
      const int reduced_col = global_to_reduced[global_col];

      if (reduced_col >= 0)
        reduced_matrix.set(reduced_row,
                           static_cast<unsigned int>(reduced_col),
                           entry->value());
    }
  }

  reduced_matrix.compress(dealii::VectorOperation::insert);
}

template class ParamSpaceContext<2, double>;
template class ParamSpaceContext<3, double>;

// // ----------------- How you use this in your adjoint/probing -----------------

// /*
// Workflow when you want FAST geometric evaluation in quadrature loops:

// 1) Once after setup:
//    ctx.compute_free_dofs();
//    ctx.build_index_maps();

// 2) Each time you have a param_free you want to evaluate many times:
//    Vector<Number> param_grid;
//    ctx.free_to_grid(param_free, param_grid, /*linear_part=*/true);

// 3) In inner loops:
//    val = ctx.evaluate_value_grid(param_grid, q_point, /*linear_part=*/true);

// This keeps your geometric interpretation AND keeps solver vectors in m_free_dofs order.
// */