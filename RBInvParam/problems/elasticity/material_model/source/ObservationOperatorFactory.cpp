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
  std::vector<Point<dim>> sensor_points;
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
  case ObservationOperatorType::Sensors:
    sensor_points = this->_get_sensor_edges(ctx);
    ObservationOperatorFactory::assemble_sensors_observation(
        ctx,
        observation_operator_matrix,
        observation_operator_sp,
        sensor_points
    );
    break;
  case ObservationOperatorType::SensorsGrid:
    sensor_points = this->_get_sensor_grids(ctx);
    ObservationOperatorFactory::assemble_sensors_observation(
        ctx,
        observation_operator_matrix,
        observation_operator_sp,
        sensor_points
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
    SparsityPattern& observation_operator_sp,
    std::vector<Point<dim>> sensor_points) const
{      
    
    double radius = std::get<double>(ctx.hyperparameter.at("radius"));
    const double tol2 = radius * radius;

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
    //const std::vector<Point<dim>> sensor_points = this->_get_sensor_points(ctx);    
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
    //observation_operator_sp.copy_from(sp_G);
    observation_operator_matrix.reinit(observation_operator_sp);
    
    G.mmult(observation_operator_matrix, boundary_mass_matrix);
}

template <int dim, typename Number>
std::vector<Point<dim>> ObservationOperatorFactory<dim, Number>::_get_sensor_edges(
    const ObservationOperatorFactoryContext<dim, Number> ctx
) const
{   
    AssertDimension(dim, 3);
    std::vector<Point<dim>> sensor_points;
    double delta_y;
    double delta_z;
    
    double y_start;
    double z_start;
    
    double y_end;
    double z_end;

    double y_pos;
    double z_pos;
    
    std::vector<double> spatial_resolution = std::get<std::vector<double>>(ctx.hyperparameter.at("spatial_resolution"));
    bool second_row = std::get<bool>(ctx.hyperparameter.at("second_row"));    
    
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

    delta_y = 30.0 / y_spatial_resolution;
    delta_z = 30.0 / z_spatial_resolution;

    y_start = -15.0 + delta_y;
    z_start = -15.0 + delta_z;
    
    y_end = 15.0 - delta_y;
    z_end = 15.0 - delta_z;

    size_t sensor_per_edge = static_cast<size_t>(y_spatial_resolution) - 2;
    sensor_points.resize(8 * sensor_per_edge);

    for (unsigned int i = 0; i < sensor_per_edge; ++i)
    {
        y_pos = y_start + i * frequence * delta_y;
        z_pos = z_start + i * frequence * delta_z;

        sensor_points[i + (0 * sensor_per_edge)] = Point<3>( 0.1, y_pos, z_start);
        sensor_points[i + (1 * sensor_per_edge)] = Point<3>( 0.1, y_pos, z_end);
        sensor_points[i + (2 * sensor_per_edge)] = Point<3>( 0.1, y_start, z_pos);
        sensor_points[i + (3 * sensor_per_edge)] = Point<3>( 0.1, y_end, z_pos);
        sensor_points[i + (4 * sensor_per_edge)] = Point<3>(-0.1, y_pos, z_start);
        sensor_points[i + (5 * sensor_per_edge)] = Point<3>(-0.1, y_pos, z_end);
        sensor_points[i + (6 * sensor_per_edge)] = Point<3>(-0.1, y_start, z_pos);
        sensor_points[i + (7 * sensor_per_edge)] = Point<3>(-0.1, y_end, z_pos);
    }

    if (!second_row) {
        return sensor_points;
    }

    y_start = -7.0;
    z_start = -7.0;
    
    y_end = 7.0;
    z_end = 7.0;

    sensor_per_edge = 14;
    
    sensor_points.resize(sensor_points.size() + 8 * sensor_per_edge);

    for (unsigned int i = 0; i < sensor_per_edge; ++i)
    {
        y_pos = y_start + i * frequence * delta_y;
        z_pos = z_start + i * frequence * delta_z;

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


template <int dim, typename Number>
std::vector<Point<dim>> ObservationOperatorFactory<dim, Number>::_get_sensor_grids(
    const ObservationOperatorFactoryContext<dim, Number> ctx
) const
{   
    AssertDimension(dim, 3);
    std::vector<Point<dim>> sensor_points;
    size_t idx;

    double x_coord;
    double y_coord;
    double z_coord;

    //std::vector<double> spatial_resolution = std::get<std::vector<double>>(ctx.hyperparameter.at("spatial_resolution"));
    std::vector<double> grid_sizes = std::get<std::vector<double>>(ctx.hyperparameter.at("grid_sizes"));
    
    sensor_points.resize(grid_sizes[0] * grid_sizes[1] * grid_sizes[2]);    
    idx = 0;

    for (unsigned int i = 0; i < grid_sizes[0]; i++) {
        x_coord = -0.1 + 0 + i * (0.2 / (grid_sizes[0] - 1));
        
        for (unsigned int j = 0; j < grid_sizes[1]; j++) {
            y_coord = -15.0 + 1 + j * (28.0 / (grid_sizes[1] - 1));
            //y_coord = -15.0 + j * (30.0 / (grid_sizes[1] - 1));
            
            for (unsigned int k = 0; k < grid_sizes[2]; k++) {
                z_coord = -15.0 + 1 + k * (28.0 / (grid_sizes[2] - 1));
                //z_coord = -15.0 + k * (30.0 / (grid_sizes[2] - 1));
                
                sensor_points[idx] = Point<3>(x_coord, y_coord, z_coord);
                idx++;

                // std::cout << "----------------------------------------" << std::endl;
                // std::cout << x_coord << std::endl;
                // std::cout << y_coord << std::endl;
                // std::cout << z_coord << std::endl;
            }
        }
    }   

    return sensor_points;
}