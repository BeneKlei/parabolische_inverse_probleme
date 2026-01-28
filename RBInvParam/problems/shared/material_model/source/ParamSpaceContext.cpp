#include "ParamSpaceContext.hpp"

using namespace dealii;

template <int dim, typename Number>
void ParamSpaceContext<dim, Number>::evaluate_values(
  const Vector<Number>          &param_full,
  const std::vector<Point<dim>> &points,
  std::vector<Number>           &values) const
{
  std::cout << "Enter" << std::endl;
  AssertDimension(param_full.size(), m_dof_handler.n_dofs());

  values.resize(points.size());
  m_rpe.reinit(points, m_dof_handler.get_triangulation(), m_mapping);

  const auto _values = VectorTools::point_values<1>(
    m_mapping,
    m_dof_handler,
    param_full,
    points,
    m_rpe
  );
  values = _values;
  std::cout << "Leave" << std::endl;
}

template <int dim, typename Number>
void ParamSpaceContext<dim, Number>::reconstruct_full_param(
  const Vector<Number>          &param,
  Vector<Number>                &param_full
) const
{
  AssertDimension(param_full.size(), m_free_dofs.size());
  
  param_full.reinit(m_dof_handler.n_dofs());

  for (std::size_t k = 0; k < m_free_dofs.size(); ++k)
    param_full[m_free_dofs[k]] = param[k];

  m_constraints.distribute(param_full);
}

// explicit instantiations (since definitions are in a .cpp)
template class ParamSpaceContext<2, double>;
template class ParamSpaceContext<3, double>;
