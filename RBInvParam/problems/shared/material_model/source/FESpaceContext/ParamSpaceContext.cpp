#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <deal.II/numerics/fe_field_function.h>

#include "FESpaceContext/ParamSpaceContext.hpp"

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