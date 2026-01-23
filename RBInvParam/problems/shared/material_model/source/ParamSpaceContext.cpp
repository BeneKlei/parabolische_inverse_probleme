#include "ParamSpaceContext.hpp"

using namespace dealii;

template <int dim, typename Number>
void ParamSpaceContext<dim, Number>::evaluate_values(
  const Vector<Number>          &param_full,
  const std::vector<Point<dim>> &points,
  std::vector<Number>           &values) const
{
  AssertThrow(m_grid_cache.get() != nullptr,
              ExcMessage("m_param_grid_cache not initialized."));

  values.resize(points.size());

  // // scalar parameter field (component 0)
  // for (std::size_t i = 0; i < points.size(); ++i)
  //   values[i] = VectorTools::point_value(m_mapping, m_dof_handler, param_full, points[i], 0);
}

template <int dim, typename Number>
void ParamSpaceContext<dim, Number>::evaluate_values_from_reduced(
  const std::vector<Number>     &param_reduced,
  Vector<Number>                &full_buffer,
  const std::vector<Point<dim>> &points,
  std::vector<Number>           &values) const
{
  AssertThrow(m_grid_cache.get() != nullptr,
              ExcMessage("m_param_grid_cache not initialized."));
  AssertDimension(param_reduced.size(), m_free_dofs.size());

  // full_buffer.reinit(m_dof_handler.n_dofs());
  // full_buffer = Number(0);

  // for (std::size_t k = 0; k < m_free_dofs.size(); ++k)
  //   full_buffer[m_free_dofs[k]] = param_reduced[k];

  // m_constraints.distribute(full_buffer);

  // evaluate_values(full_buffer, points, values);
}

// explicit instantiations (since definitions are in a .cpp)
template class ParamSpaceContext<2, double>;
template class ParamSpaceContext<3, double>;
