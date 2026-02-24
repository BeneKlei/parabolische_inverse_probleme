#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

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

template <int dim, typename Number>
Number ParamSpaceContext<dim, Number>::evaluate_value(
  const dealii::Vector<Number> &param,
  const dealii::Point<dim>     &point,
  bool                          linear_part) const
{
  // Here param is the FREE vector
  AssertDimension(param.size(), m_free_dofs.size());

  const unsigned int Ny = m_grid_resolution[1];
  const unsigned int Nz = m_grid_resolution[2];

  // You used stride = Ny+1; keep your convention
  const unsigned int stride = Ny + 1;

  // If x != p1[0], you return fixed value (affine boundary / extrusion)
  if (point[0] != m_p1[0])
  {
    return linear_part ? Number(0.0) : Number(1.0);
  }

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

  return Number(param[static_cast<unsigned int>(iy) +
                      static_cast<unsigned int>(iz) * stride]);
}

template <int dim, typename Number>
void ParamSpaceContext<dim, Number>::evaluate_values(
  const dealii::Vector<Number>          &param,
  const std::vector<dealii::Point<dim>> &points,
  std::vector<Number>                   &values,
  bool                                   linear_part) const
{
  AssertDimension(param.size(), m_free_dofs.size());
  values.resize(points.size());

  for (std::size_t i = 0; i < points.size(); ++i)
    values[i] = evaluate_value(param, points[i], linear_part);
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


// explicit instantiations (since definitions are in a .cpp)
template class ParamSpaceContext<2, double>;
template class ParamSpaceContext<3, double>;