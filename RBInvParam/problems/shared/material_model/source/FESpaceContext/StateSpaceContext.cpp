#include <deal.II/fe/fe_values.h>

#include "FESpaceContext/StateSpaceContext.hpp"

template <int dim, typename Number>
void StateSpaceContext<dim, Number>::pre_compute()
{
  pre_compute_quad_points_flat();
  m_pre_computed = true;
}

template <int dim, typename Number>
void StateSpaceContext<dim, Number>::pre_compute_quad_points_flat()
{
  m_quad_points_flat.clear();

  dealii::FEValues<dim> fe_values_state(
      this->m_mapping,
      this->m_fe,
      this->m_quadrature,
      dealii::update_quadrature_points);

  for (const auto &cell : this->m_dof_handler.active_cell_iterators())
  {
    fe_values_state.reinit(cell);  // IMPORTANT

    const auto &qpts = fe_values_state.get_quadrature_points();
    m_quad_points_flat.insert(m_quad_points_flat.end(), qpts.begin(), qpts.end());
  }
}

// explicit instantiations (since definitions are in a .cpp)
template class StateSpaceContext<2, double>;
template class StateSpaceContext<3, double>;