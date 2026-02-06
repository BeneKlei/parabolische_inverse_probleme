#include <deal.II/numerics/fe_field_function.h>

#include "ParamSpaceContext.hpp"

using namespace dealii;

// template <int dim, typename Number>
// void ParamSpaceContext<dim, Number>::evaluate_values(
//   const Vector<Number>          &param_full,
//   const std::vector<Point<dim>> &points,
//   std::vector<Number>           &values) const
// {
//   std::cout << "Enter" << std::endl;
//   AssertDimension(param_full.size(), m_dof_handler.n_dofs());

//   values.resize(points.size());
//   //m_rpe.reinit(points, m_dof_handler.get_triangulation(), m_mapping);
//   std::cout << "Set up" << std::endl;

//   const auto _values = VectorTools::point_values<1>(
//     m_mapping,
//     m_dof_handler,
//     param_full,
//     points,
//     m_rpe
//   );
//   values = _values;
//   std::cout << "Leave" << std::endl;
// }

template <int dim, typename Number>
Number ParamSpaceContext<dim, Number>::evaluate_value(
  const Vector<Number>  &param,
  const Point<dim>      &point,
  bool                   linear_part) const
{
  AssertDimension(param.size(), m_free_dofs.size());
  
  const unsigned int Ny = m_grid_resolution[1];
  const unsigned int Nz = m_grid_resolution[2];
  const unsigned int stride = Nz + 1;

  if (point[0] != m_p1[0]) 
  { 
    if (linear_part) {
      return Number(0.0); 
    }
    else
    {
      return Number(1.0); 
    }
  }

  const double y_min = m_p1[1] , y_max = m_p2[1];
  const double z_min = m_p1[2] , z_max = m_p2[1];

  auto clamp = [](int v, int lo, int hi){ return std::max(lo, std::min(v, hi)); };

  int iy = clamp((int)std::lround((point[1]-y_min)/(y_max-y_min) * Ny), 0, (int)Ny);
  int iz = clamp((int)std::lround((point[2]-z_min)/(z_max-z_min) * Nz), 0, (int)Nz);

  return Number(param[iy * stride + iz]);  
}


template <int dim, typename Number>
void ParamSpaceContext<dim, Number>::evaluate_values(
  const Vector<Number>          &param,
  const std::vector<Point<dim>> &points,
  std::vector<Number>           &values,
  bool                           linear_part) const
{
  AssertDimension(param.size(), m_free_dofs.size());
  values.resize(points.size());

  for (std::size_t i=0; i<points.size(); ++i)
  {    
    values[i] = evaluate_value(param,
                               points[i],
                               linear_part);
  }
}

template <int dim, typename Number>
void ParamSpaceContext<dim, Number>::project_to_free_param(
  const Vector<Number> &full_param,
  Vector<Number>       &free_param) const
{
  AssertDimension(full_param.size(), m_dof_handler.n_dofs());
  free_param.reinit(m_free_dofs.size()); 

  unsigned int fi = 0;
  for (types::global_dof_index gi = 0; gi < m_dof_handler.n_dofs(); ++gi)
    if (!m_constraints.is_constrained(gi))
    {
      AssertIndexRange(fi, free_param.size());
      free_param[fi++] = full_param[gi];
    }

  AssertDimension(fi, free_param.size());
}

// template <int dim, typename Number>
// void ParamSpaceContext<dim, Number>::reconstruct_full_param(
//   const Vector<Number>          &param,
//   Vector<Number>                &param_full
// ) const
// {
//   AssertDimension(param_full.size(), m_free_dofs.size());
  
//   param_full.reinit(m_dof_handler.n_dofs());

//   for (std::size_t k = 0; k < m_free_dofs.size(); ++k)
//     param_full[m_free_dofs[k]] = param[k];

//   m_constraints.distribute(param_full);
// }

// explicit instantiations (since definitions are in a .cpp)
template class ParamSpaceContext<2, double>;
template class ParamSpaceContext<3, double>;
