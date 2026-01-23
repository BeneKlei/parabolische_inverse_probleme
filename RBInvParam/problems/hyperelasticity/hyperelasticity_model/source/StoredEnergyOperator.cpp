#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include "StoredEnergyOperator.hpp"

using namespace dealii;

template <int dim, typename Number>
StoredEnergyOperator<dim, Number>::StoredEnergyOperator(
    const Vector<Number>                    &q,
    const StateSpaceContext<dim>            &state_space_context,
    const ParamSpaceContext<dim, Number>    &param_space_context,
    const StoredEnergyFunction<dim, Number> &stored_energy_function)
  : m_q(q)
  , m_state_space_context(state_space_context)
  , m_param_space_context(param_space_context)
  , m_stored_energy_function(stored_energy_function)
{}

template <int dim, typename Number>
std::size_t StoredEnergyOperator<dim, Number>::dim_source() const
{
  return m_state_space_context.dof_handler().n_dofs();
}

template <int dim, typename Number>
std::size_t StoredEnergyOperator<dim, Number>::dim_range() const
{
  return m_state_space_context.dof_handler().n_dofs();
}

template <int dim, typename Number>
void StoredEnergyOperator<dim, Number>::apply(Vector<Number>       &y,
                                              const Vector<Number> &u) const
{
  // --- quadrature ---
  QGaussLobatto<dim> quadrature_formula(2);
  const unsigned int n_q = quadrature_formula.size();

  // --- FEValues for state ---
  FEValues<dim> fe_values_state(m_state_space_context.fe(),
                                quadrature_formula,
                                update_gradients | update_JxW_values |
                                update_quadrature_points | update_values);

  const unsigned int dofs_per_cell = m_state_space_context.fe().dofs_per_cell;

  // --- local storage ---
  Vector<Number> local_y(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const FEValuesExtractors::Vector vel(0);

  std::vector<Tensor<2, dim>> u_gradients(n_q);
  std::vector<Point<dim>>     q_points(n_q);
  std::vector<Number> param_values(n_q);
  std::vector<Tensor<2, dim>> DY_stored_energy_points(n_q);

  const Tensor<2, dim> I = Tensor<2, dim>(unit_symmetric_tensor<dim, Number>());

  y = Number(0);

  for (const auto &cell : m_state_space_context.dof_handler().active_cell_iterators())
  {
    local_y = Number(0);

    fe_values_state.reinit(cell);

    fe_values_state[vel].get_function_gradients(u, u_gradients);
    q_points = fe_values_state.get_quadrature_points();

    for (unsigned int q = 0; q < n_q; ++q)
    {      
      DY_stored_energy_points[q] =
          m_stored_energy_function.gradient(q_points[q], u_gradients[q] + I);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const unsigned int component_i =
            m_state_space_context.fe().system_to_component_index(i).first;

        local_y(i) += DY_stored_energy_points[q][component_i] *
                      fe_values_state.shape_grad(i, q) *
                      fe_values_state.JxW(q);
      }
    }

    cell->get_dof_indices(local_dof_indices);
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      y(local_dof_indices[i]) += local_y(i);
  }
}


template <int dim, typename Number>
std::unique_ptr<BaseOperator<Number>> 
StoredEnergyOperator<dim, Number>::jacobian(const Vector<Number> &u) const
{
  // --- quadrature ---
  QGaussLobatto<dim> quadrature_formula(2);
  const unsigned int n_q = quadrature_formula.size();

  // --- FEValues for state ---
  FEValues<dim> fe_values_state(m_state_space_context.fe(),
                                quadrature_formula,
                                update_gradients | update_JxW_values |
                                update_quadrature_points | update_values);
  
  const unsigned int dofs_per_cell = m_state_space_context.fe().dofs_per_cell;

  // --- local storage ---
  FullMatrix<Number> local_J(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const FEValuesExtractors::Vector vel(0);

  std::vector<Tensor<2, dim>> u_gradients(n_q);
  std::vector<Point<dim>>     q_points(n_q);
  std::vector<Number> param_values(n_q);
  std::vector<Tensor<2, dim>> DY_DY_H_stored_energy_points(n_q);

  Tensor<2,dim> test_j_H;

  const Tensor<2, dim> I = Tensor<2, dim>(unit_symmetric_tensor<dim, Number>());
  
  SparseMatrix<Number> J;
  J.reinit(m_state_space_context.state_sp());
  J = Number(0);

  for (const auto &cell : m_state_space_context.dof_handler().active_cell_iterators())
  {
    local_J = Number(0);
    fe_values_state.reinit(cell);

    fe_values_state[vel].get_function_gradients(u, u_gradients);
    q_points = fe_values_state.get_quadrature_points();

    // TODO 

    for (unsigned int q = 0; q < n_q; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const unsigned int component_i = 
          m_state_space_context.fe().system_to_component_index(i).first;
        
        DY_DY_H_stored_energy_points[q] =
          m_stored_energy_function.contracted_hessian(
            q_points[q], 
            u_gradients[q] + I,
            fe_values_state.shape_grad(i, q),
            component_i
        );

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          const unsigned int component_j = 
            m_state_space_context.fe().system_to_component_index(j).first;
          
          test_j_H.clear();
          test_j_H[component_j] = fe_values_state.shape_grad(j, q);

          local_J(i, j) += double_contract<0,0,1,1>(DY_DY_H_stored_energy_points[q], test_j_H) * 
                           fe_values_state.JxW(q);
        }
      }
    }
  }

  return std::make_unique<SparseMatrixOperator<Number>>(
    J
  );
}



template class StoredEnergyOperator<2, double>;
template class StoredEnergyOperator<3, double>;


