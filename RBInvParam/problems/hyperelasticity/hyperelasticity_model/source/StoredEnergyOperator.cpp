#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include "StoredEnergyOperator.hpp"

template <int dim, typename Number>
StoredEnergyOperator<dim, Number>::StoredEnergyOperator(
    const StoredEnergyFunction<dim, Number>& stored_energy_function, 
    const StoredEnergyOperatorContext<dim>& ctx
)   
    : m_stored_energy_function(stored_energy_function)
    , m_ctx(ctx)
{}

template <int dim, typename Number>
void StoredEnergyOperator<dim, Number>::apply(Vector<Number>       &y,
                                              const Vector<Number> &u) const
{
    QGaussLobatto<3> quadrature_formula(2);
    FEValues<dim> fe_values(m_ctx.fe, quadrature_formula,
                            update_gradients | update_JxW_values | update_quadrature_points | update_values);
    const unsigned int dofs_per_cell       = m_ctx.fe.dofs_per_cell;
    const unsigned int n_quadrature_points = quadrature_formula.size();

    Vector<double> local_y(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const FEValuesExtractors::Vector vel(0);

    std::vector<Tensor<2,dim>> u_gradients(n_quadrature_points);   
    std::vector<Point<dim>> q_points_coords(n_quadrature_points);
    std::vector<Tensor<2,dim>> DY_stored_energy_points(n_quadrature_points);

    Tensor<2,dim> I = Tensor<2, dim>(unit_symmetric_tensor<dim, Number>());
    
    y = Number(0);

    for (const auto &cell : m_ctx.dof_handler.active_cell_iterators())
    {   
        local_y = Number(0);
        fe_values.reinit(cell);

        fe_values[vel].get_function_gradients(u, u_gradients);
        q_points_coords = fe_values.get_quadrature_points();

        for (unsigned int q_point=0; q_point<n_quadrature_points; ++q_point){

            DY_stored_energy_points[q_point] = m_stored_energy_function.value(
                q_points_coords[q_point],
                u_gradients[q_point] + I
            );
            
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                const unsigned int component_i = m_ctx.fe.system_to_component_index(i).first;
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