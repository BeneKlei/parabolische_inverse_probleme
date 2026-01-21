#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include "StoredEnergyOperator.hpp"

template <int dim, typename Number>
StoredEnergyOperator<dim, Number>::StoredEnergyOperator(
    const StoredEnergyFunction<dim, Number>& stored_energy_function,
    const FiniteElement<dim>& fe,
    const DoFHandler<dim>& dof_handler
)   
    : m_stored_energy_function(stored_energy_function)
    , m_fe(fe)
    , m_dof_handler(dof_handler)
{}

template <int dim, typename Number>
std::size_t StoredEnergyOperator<dim, Number>::dim_source() const
{
    return m_dof_handler.n_dofs();
}

template <int dim, typename Number>
std::size_t StoredEnergyOperator<dim, Number>::dim_range() const
{
    return m_dof_handler.n_dofs();
}

template <int dim, typename Number>
void StoredEnergyOperator<dim, Number>::apply(Vector<Number>       &y,
                                              const Vector<Number> &u) const
{
    QGaussLobatto<3> quadrature_formula(2);
    FEValues<dim> fe_values(m_fe, quadrature_formula,
                            update_gradients | update_JxW_values | update_quadrature_points | update_values);
    const unsigned int dofs_per_cell       = m_fe.dofs_per_cell;
    const unsigned int n_quadrature_points = quadrature_formula.size();

    Vector<double> local_y(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const FEValuesExtractors::Vector vel(0);

    std::vector<Tensor<2,dim>> u_gradients(n_quadrature_points);   
    std::vector<Point<dim>> q_points_coords(n_quadrature_points);
    std::vector<Tensor<2,dim>> DY_stored_energy_points(n_quadrature_points);

    Tensor<2,dim> I = Tensor<2, dim>(unit_symmetric_tensor<dim, Number>());
    
    y = Number(0);

    for (const auto &cell : m_dof_handler.active_cell_iterators())
    {   
        local_y = Number(0);
        fe_values.reinit(cell);

        fe_values[vel].get_function_gradients(u, u_gradients);
        q_points_coords = fe_values.get_quadrature_points();

        for (unsigned int q_point=0; q_point<n_quadrature_points; ++q_point){

            DY_stored_energy_points[q_point] = m_stored_energy_function->gradient(
                q_points_coords[q_point],
                u_gradients[q_point] + I
            );
            
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                const unsigned int component_i = m_fe.system_to_component_index(i).first;
                local_y(i) += DY_stored_energy_points[q_point][component_i] *
                    fe_values.shape_grad(i,q_point) * fe_values.JxW(q_point);
            }
        }

        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            y(local_dof_indices[i]) += local_y(i);
        }
    }
}

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