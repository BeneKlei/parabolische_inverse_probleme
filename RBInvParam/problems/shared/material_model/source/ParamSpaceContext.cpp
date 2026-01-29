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


// template <int dim, typename Number>
// void ParamSpaceContext<dim, Number>::evaluate_values(
//   const Vector<Number>          &param_full,
//   const std::vector<Point<dim>> &points,
//   std::vector<Number>           &values) const
// {
//   std::cout << "Enter" << std::endl;
//   AssertDimension(param_full.size(), m_dof_handler.n_dofs());

//   values.resize(points.size());
//   Functions::FEFieldFunction<dim> fe_function(
//     m_dof_handler, 
//     param_full,
//     m_mapping
//   );
//   std::cout << "Set up" << std::endl;

//   fe_function.value_list(points, values);

//   std::cout << "Leave" << std::endl;
// }



// TODO This implementation is maximal quick and dirty and should be replaced later!
template <int dim, typename Number>
void ParamSpaceContext<dim, Number>::evaluate_values(
  const Vector<Number>          &param_full,
  const std::vector<Point<dim>> &points,
  std::vector<Number>           &values) const
{
  std::cout << "Enter" << std::endl;

  AssertDimension(param_full.size(), m_dof_handler.n_dofs());
  
  //m_rpe.reinit(points, m_dof_handler.get_triangulation(), m_mapping);
  values.resize(points.size());
  const Number threshold = Number(0.2) / Number(30);
          
  std::cout << "Set up" << std::endl;
  
  for (std::size_t i = 0; i < points.size(); ++i)
  {
    const auto &p = points[i];
    // values[i] = (p[0] >= threshold) ? Number(1.0) : 
    // VectorTools::point_value(
    //   m_mapping,
    //   m_dof_handler,
    //   param_full,
    //   p
    // );
  }

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
