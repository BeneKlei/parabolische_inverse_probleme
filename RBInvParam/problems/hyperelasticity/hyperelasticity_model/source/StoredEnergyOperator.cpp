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

  // param factor at q-points (scalar field assumed)
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

    // // ---- evaluate parameter field at the state quadrature points ----
    // // IMPORTANT:
    // // This assumes `m_q` is a FULL parameter vector in param DoF indexing.
    // // If m_q is reduced (free dofs only), use evaluate_values_from_reduced(...)
    // // and pass in a reusable full_buffer.
    // {
    //   // Convert deal.II Vector<Number> -> std::vector<Number> view/copy:
    //   std::vector<Number> q_full(m_q.size());
    //   for (unsigned int i = 0; i < m_q.size(); ++i)
    //     q_full[i] = m_q[i];

    //   m_param_space_context.evaluate_values(q_full, q_points, param_values);
    // }

    for (unsigned int q = 0; q < n_q; ++q)
    {
      // Here you can use param_values[q] as factor if your energy depends on it.
      // For now I keep your original call signature; adapt as needed.
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

template class StoredEnergyOperator<2, double>;
template class StoredEnergyOperator<3, double>;
// namespace dealii_like
// {
//   using namespace dealii;

//   template <int dim, typename Number>
//   void StoredEnergyOperator<dim, Number>::build_first_layer_flags()
//   {
//     const auto &tria = m_state_dh.get_triangulation();
//     m_is_first_layer.assign(tria.n_active_cells(), 0);

//     for (const auto &cell : m_state_dh.active_cell_iterators())
//     {
//       bool touches = false;
//       for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
//         if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 1)
//         {
//           touches = true;
//           break;
//         }

//       m_is_first_layer[cell->active_cell_index()] = touches ? 1 : 0;
//     }
//   }

//   template <int dim, typename Number>
//   void StoredEnergyOperator<dim, Number>::apply(Vector<Number>       &y,
//                                                 const Vector<Number> &u) const
//   {
//     // --- quadrature ---
//     QGaussLobatto<dim> quadrature_formula(2);
//     const unsigned int n_q = quadrature_formula.size();

//     // --- FEValues for state and for param ---
//     FEValues<dim> fe_values_state(m_state_fe, quadrature_formula,
//                                   update_gradients | update_JxW_values);

//     FEValues<dim> fe_values_param(m_param_fe, quadrature_formula,
//                                   update_values);

//     const unsigned int dofs_per_cell = m_state_fe.dofs_per_cell;

//     // --- local storage (reused per cell) ---
//     Vector<Number> local_y(dofs_per_cell);
//     std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

//     // vector-valued displacement/velocity field in component block 0
//     const FEValuesExtractors::Vector vel(0);

//     std::vector<Tensor<2, dim>> u_gradients(n_q);
//     std::vector<double>         param_factor_q(n_q);

//     // deformation gradient F = I + grad(u)
//     const Tensor<2, dim, Number> I = unit_symmetric_tensor<dim, Number>();

//     y = Number(0);

//     for (const auto &cell : m_state_dh.active_cell_iterators())
//     {
//       local_y = Number(0);

//       fe_values_state.reinit(cell);
//       fe_values_state[vel].get_function_gradients(u, u_gradients);

//       // Map state cell -> parameter cell (requires same triangulation and cell indexing)
//       auto cell_param =
//         typename DoFHandler<dim>::active_cell_iterator(&cell->get_triangulation(),
//                                                        cell->level(),
//                                                        cell->index(),
//                                                        &m_param_dh);

//       fe_values_param.reinit(cell_param);

//       // param_factor_q[q] = param_solution evaluated at q-point
//       // If your parameter is not scalar or not component 0, adapt this call accordingly.
//       fe_values_param.get_function_values(m_param_solution, param_factor_q);

//       const bool first_layer = (m_is_first_layer[cell->active_cell_index()] != 0);

//       for (unsigned int q = 0; q < n_q; ++q)
//       {
//         const double pf = first_layer ? param_factor_q[q] : 1.0;

//         const Tensor<2, dim, Number> F = u_gradients[q] + I;

//         // First Piola stress P = dW/dF (with factor)
//         const Tensor<2, dim, Number> P = m_stored_energy.gradient_with_factor(pf, F);

//         for (unsigned int i = 0; i < dofs_per_cell; ++i)
//         {
//           // For a vector-valued FE, gradient(i,q) gives grad(phi_i) as Tensor<2,dim>
//           const Tensor<2, dim, Number> grad_phi_i = fe_values_state[vel].gradient(i, q);

//           // Add (P : grad_phi_i) * JxW
//           local_y(i) += scalar_product(P, grad_phi_i) * fe_values_state.JxW(q);
//         }
//       }

//       cell->get_dof_indices(local_dof_indices);
//       for (unsigned int i = 0; i < dofs_per_cell; ++i)
//         y(local_dof_indices[i]) += local_y(i);
//     }
//   }

//   // explicit instantiations (adjust as you need)
//   template class StoredEnergyOperator<2, double>;
//   template class StoredEnergyOperator<3, double>;

// } // namespace dealii_like