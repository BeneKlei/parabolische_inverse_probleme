#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>

#include "ObservationOperatorFactory.hpp"
#include "utils.hpp"

template class ObservationOperatorFactory<3, double>;

template <int dim, typename Number>
void ObservationOperatorFactory<dim, Number>::assemble_observation(
    ObservationOperatorFactoryContext<dim, Number> ctx,
    SparseMatrix<Number>& observation_operator_matrix,
    SparsityPattern& observation_operator_sp) const
{
  switch (ctx.observation_operator_type)
  {
  case ObservationOperatorType::Identity:
    ObservationOperatorFactory::assemble_identity_observation(
        ctx,
        observation_operator_matrix,
        observation_operator_sp
    );  
    break;
  case ObservationOperatorType::Boundary:
    ObservationOperatorFactory::assemble_boundary_observation(
        ctx,
        observation_operator_matrix,
        observation_operator_sp
    );
    break;
  case ObservationOperatorType::SensorsR8d:
  case ObservationOperatorType::SensorsR9d:
    ObservationOperatorFactory::assemble_sensors_observation(
        ctx,
        observation_operator_matrix,
        observation_operator_sp
    );
    break;
  default:
    throw std::runtime_error("Unknown ObservationOperatorType.");
  }
}

template <int dim, typename Number>
void ObservationOperatorFactory<dim, Number>::assemble_identity_observation(
    const ObservationOperatorFactoryContext<dim, Number> ctx,
    SparseMatrix<Number>& observation_operator_matrix,
    SparsityPattern& observation_operator_sp) const
{   
    observation_operator_sp.copy_from(ctx.sparsity_pattern);
    observation_operator_matrix.reinit(observation_operator_sp);
    observation_operator_matrix = 0;

    for (types::global_dof_index i = 0; i < ctx.dof_handler.n_dofs(); ++i)
        observation_operator_matrix.set(i, i, Number(1));
    
}

template <int dim, typename Number>
void ObservationOperatorFactory<dim, Number>::assemble_boundary_observation(
    const ObservationOperatorFactoryContext<dim, Number> ctx,
    SparseMatrix<Number>& observation_operator_matrix,
    SparsityPattern& observation_operator_sp) const
{
    StateProductFactoryContext<3, Number> boundary_mass_ctx {
        StateProductType::BoundaryMass,
        ctx.fe,
        ctx.dof_handler,
        ctx.sparsity_pattern    
    };

    observation_operator_sp.copy_from(ctx.sparsity_pattern);
    observation_operator_matrix.reinit(observation_operator_sp);

    m_state_product_factory.assemble_state_product(
        boundary_mass_ctx,
        observation_operator_matrix
    );
   
}

template <int dim, typename Number>
void ObservationOperatorFactory<dim, Number>::assemble_sensors_observation(
    const ObservationOperatorFactoryContext<dim, Number> ctx,
    SparseMatrix<Number>& observation_operator_matrix,
    SparsityPattern& observation_operator_sp) const
{      
    const Number tol  = 5 * 1e0;
    const Number tol2 = tol * tol;

    SparseMatrix<Number> G;
    SparseMatrix<Number> boundary_mass_matrix;

    StateProductFactoryContext<3, Number> boundary_mass_ctx {
        StateProductType::BoundaryMass,
        ctx.fe,
        ctx.dof_handler,
        ctx.sparsity_pattern    
    };

    m_state_product_factory.assemble_state_product(
        boundary_mass_ctx,
        boundary_mass_matrix
    );
    const std::vector<Point<dim>> sensor_points = this->_get_sensor_points(ctx.observation_operator_type);    
    const unsigned int L = ctx.dof_handler.n_dofs();
    const unsigned int l = sensor_points.size();
    std::vector<std::vector<types::global_dof_index>> rows(l);

    // ----------------------------------------------------

    MappingQ1<dim> mapping;
    std::vector<Point<dim>> support_points(ctx.dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(mapping, ctx.dof_handler, support_points);

    for (unsigned int si = 0; si < l; ++si)
    {
        for (unsigned int i = 0; i < support_points.size(); ++i)
        {   
            if ((sensor_points[si] - support_points[i]).norm_square() <= tol2 ) {
                rows[si].push_back(i);
            }
        }
    }
    
    // ----------------------------------------------------

    DynamicSparsityPattern dsp(l, L);
    for (unsigned int i = 0; i < l; ++i)
    {
        for (auto dof : rows[i]) 
        {
            dsp.add(i, dof);
        }
            
    }
        
    SparsityPattern sp_G;
    sp_G.copy_from(dsp);

    G.reinit(sp_G);
    for (unsigned int i = 0; i < l; ++i)
    {   
        for (auto dof : rows[i]) 
        {      
            G.set(i, dof, 1.0);
        }
            
    }
    // ----------------------------------------------------

    observation_operator_sp.copy_from(utils::make_product_sparsity_AB(G, boundary_mass_matrix));
    observation_operator_matrix.reinit(observation_operator_sp);
    
    G.mmult(observation_operator_matrix, boundary_mass_matrix);
}

template <int dim, typename Number>
std::vector<Point<dim>> ObservationOperatorFactory<dim, Number>::_get_sensor_points(
    const ObservationOperatorType &observation_operator_type
) const
{   
    AssertDimension(dim, 3);

    std::vector<Point<dim>> sensor_points;

    switch (observation_operator_type) {
    case ObservationOperatorType::SensorsR9d:
        sensor_points.resize(64);
        for (unsigned int i = 0; i < 8; i++) {
            sensor_points[i] = Point<3>(0.1, -16.0 + i*4.0, -16.0);
            sensor_points[i + 8] = Point<3>(0.1, 16.0, -16.0 + i*4.0);
            sensor_points[i + 16] = Point<3>(0.1, 16.0 - i*4.0, 16.0);
            sensor_points[i + 24] = Point<3>(0.1, -16.0, 16.0 - i*4.0);
            sensor_points[i + 32] = Point<3>(-0.1, -16.0 + i*4.0, -16.0);
            sensor_points[i + 40] = Point<3>(-0.1, 16.0, -16.0 + i*4.0);
            sensor_points[i + 48] = Point<3>(-0.1, 16.0 - i*4.0, 16.0);
            sensor_points[i + 56] = Point<3>(-0.1, -16.0, 16.0 - i*4.0);
        }
        break;
   case ObservationOperatorType::SensorsR8d:
        sensor_points.resize(56);
        for (unsigned int i = 0; i < 7; i++) {
            sensor_points[i] = Point<3>(0.1, -14.0 + i*4.0, -14.0);
            sensor_points[i + 7] = Point<3>(0.1, 14.0, -14.0 + i*4.0);
            sensor_points[i + 14] = Point<3>(0.1, 14.0 - i*4.0, 14.0);
            sensor_points[i + 21] = Point<3>(0.1, -14.0, 14.0 - i*4.0);
            sensor_points[i + 28] = Point<3>(-0.1, -14.0 + i*4.0, -14.0);
            sensor_points[i + 35] = Point<3>(-0.1, 14.0, -14.0 + i*4.0);
            sensor_points[i + 42] = Point<3>(-0.1, 14.0 - i*4.0, 14.0);
            sensor_points[i + 49] = Point<3>(-0.1, -14.0, 14.0 - i*4.0);
        }
        break;
    default:
        throw std::runtime_error(
            "ObservationOperatorType is unknown."
        );
    }
    return sensor_points;
}