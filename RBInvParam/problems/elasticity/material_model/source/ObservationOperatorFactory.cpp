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
//   case ObservationOperatorType::SensorsR8d:
//   case ObservationOperatorType::SensorsR9d:
  case ObservationOperatorType::Sensors:
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
    const Number tol  = 1e-3;
    //const Number tol  = 1e0;
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
    const std::vector<Point<dim>> sensor_points = this->_get_sensor_points(ctx);    
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
    const ObservationOperatorFactoryContext<dim, Number> ctx
) const
{   
    AssertDimension(dim, 3);
    std::vector<Point<dim>> sensor_points;
    
    std::vector<double> spatial_resolution = std::get<std::vector<double>>(ctx.hyperparameter.at("spatial_resolution"));
    //double frequence  = std::get<double>(ctx.hyperparameter.at("frequence"));
    double frequence = 1.0;

    double y_spatial_resolution = spatial_resolution[1];
    double z_spatial_resolution = spatial_resolution[2];
    
    AssertThrow(y_spatial_resolution == z_spatial_resolution,
             ExcMessage("y and z spatial resolutions must be equal"));
    AssertThrow(std::fmod(y_spatial_resolution, 2.0) == 0.0,
             ExcMessage("y_spatial_resolution must be divisible by 2"));
    AssertThrow(std::fmod(z_spatial_resolution, 2.0) == 0.0,
             ExcMessage("z_spatial_resolution must be divisible by 2"));

    double delta_y = 30.0 / y_spatial_resolution;
    double delta_z = 30.0 / z_spatial_resolution;
    
    double y_start = -15.0 + delta_y;
    double z_start = -15.0 + delta_z;
    
    double y_end = 15.0 - delta_y;
    double z_end = 15.0 - delta_z;

    size_t sensor_per_edge = static_cast<size_t>(y_spatial_resolution) - 2;
    sensor_points.resize(8 * sensor_per_edge);

    for (unsigned int i = 0; i < sensor_per_edge; ++i)
    {
        double y_pos = y_start + i * frequence * delta_y;
        double z_pos = z_start + i * frequence * delta_z;

        sensor_points[i + (0 * sensor_per_edge)] = Point<3>( 0.1, y_pos, z_start);
        sensor_points[i + (1 * sensor_per_edge)] = Point<3>( 0.1, y_pos, z_end);
        sensor_points[i + (2 * sensor_per_edge)] = Point<3>( 0.1, y_start, z_pos);
        sensor_points[i + (3 * sensor_per_edge)] = Point<3>( 0.1, y_end, z_pos);
        sensor_points[i + (4 * sensor_per_edge)] = Point<3>(-0.1, y_pos, z_start);
        sensor_points[i + (5 * sensor_per_edge)] = Point<3>(-0.1, y_pos, z_end);
        sensor_points[i + (6 * sensor_per_edge)] = Point<3>(-0.1, y_start, z_pos);
        sensor_points[i + (7 * sensor_per_edge)] = Point<3>(-0.1, y_end, z_pos);
    }

    
    return sensor_points;
}