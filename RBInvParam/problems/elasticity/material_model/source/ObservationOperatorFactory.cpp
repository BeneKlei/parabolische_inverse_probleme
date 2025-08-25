#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>

#include "ObservationOperatorFactory.hpp"
#include "utils.hpp"

template class ObservationOperatorFactory<3, double>;

template <int dim, typename Number>
void ObservationOperatorFactory<dim, Number>::assemble_observation(
    const ObservationOperatorType &observation_operator_type,
    const FiniteElement<dim> &fe,
    const DoFHandler<dim> &dof_handler,
    const AffineConstraints<Number> &constraints,
    const SparsityPattern &sparsity_pattern,
    SparseMatrix<Number>& observation_operator_matrix,
    SparsityPattern& observation_operator_sp) const
{
  switch (observation_operator_type)
  {
  case ObservationOperatorType::Identity:
    std::cout << "\t Using Identity ObservationOperator" << std::endl;
    ObservationOperatorFactory::assemble_identity_observation(
        observation_operator_type,
        fe,
        dof_handler,
        constraints,
        sparsity_pattern,
        observation_operator_matrix,
        observation_operator_sp
    );  
    break;
  case ObservationOperatorType::Boundary:
    std::cout << "\t Using Boundary ObservationOperator" << std::endl;
    ObservationOperatorFactory::assemble_boundary_observation(
        observation_operator_type,
        fe,
        dof_handler,
        constraints,
        sparsity_pattern,
        observation_operator_matrix,
        observation_operator_sp
    );
    break;
  case ObservationOperatorType::SensorsR8d:
  case ObservationOperatorType::SensorsR9d:
    std::cout << "\t Using Sensors ObservationOperator" << std::endl;
    ObservationOperatorFactory::assemble_sensors_observation(
        observation_operator_type,
        fe,
        dof_handler,
        constraints,
        sparsity_pattern,
        observation_operator_matrix,
        observation_operator_sp
    );
    break;
  default:
    throw std::runtime_error("Unknown system matrix type.");
  }
}

template <int dim, typename Number>
void ObservationOperatorFactory<dim, Number>::assemble_identity_observation(
    const ObservationOperatorType &observation_operator_type,
    const FiniteElement<dim> &fe,
    const DoFHandler<dim> &dof_handler,
    const AffineConstraints<Number> &constraints,
    const SparsityPattern &sparsity_pattern,
    SparseMatrix<Number>& observation_operator_matrix,
    SparsityPattern& observation_operator_sp) const
{
    observation_operator_matrix.reinit(sparsity_pattern);
    observation_operator_matrix = 0;

    for (types::global_dof_index i = 0; i < dof_handler.n_dofs(); ++i)
        observation_operator_matrix.set(i, i, Number(1));
    
    observation_operator_sp = sparsity_pattern;
}

template <int dim, typename Number>
void ObservationOperatorFactory<dim, Number>::assemble_boundary_observation(
    const ObservationOperatorType &observation_operator_type,
    const FiniteElement<dim> &fe,
    const DoFHandler<dim> &dof_handler,
    const AffineConstraints<Number> &constraints,
    const SparsityPattern &sparsity_pattern,
    SparseMatrix<Number>& observation_operator_matrix,
    SparsityPattern& observation_operator_sp) const
{
    ObservationOperatorFactory<dim, Number>::_assemble_boundary_mass_matrix(
        fe,
        dof_handler,
        constraints,
        sparsity_pattern,
        observation_operator_matrix);
    observation_operator_sp = sparsity_pattern;
}

template <int dim, typename Number>
void ObservationOperatorFactory<dim, Number>::assemble_sensors_observation(
    const ObservationOperatorType &observation_operator_type,
    const FiniteElement<dim> &fe,
    const DoFHandler<dim> &dof_handler,
    const AffineConstraints<Number> &constraints,
    const SparsityPattern &sparsity_pattern,
    SparseMatrix<Number>& observation_operator_matrix,
    SparsityPattern& observation_operator_sp) const
{      
    const Number tol  = 1e-8;
    const Number tol2 = tol * tol;

    SparseMatrix<Number> G;
    SparseMatrix<Number> boundary_mass_matrix;
    this->_assemble_boundary_mass_matrix(
        fe,
        dof_handler,
        constraints,
        sparsity_pattern,
        boundary_mass_matrix
    );
    const std::vector<Point<dim>> sensor_points = this->_get_sensor_points(observation_operator_type);    
    const unsigned int L = dof_handler.n_dofs();
    const unsigned int l = sensor_points.size();
    std::vector<std::vector<types::global_dof_index>> rows(l);

    // ----------------------------------------------------

    MappingQ1<dim> mapping;
    std::vector<Point<dim>> support_points(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

    for (unsigned int si = 0; si < l; ++si)
    {
        for (unsigned int i = 0; i < support_points.size(); ++i)
        {
            if ( (sensor_points[si] - support_points[i]).norm_square() <= tol2 )
                rows[si].push_back(i);
        }
    }
    
    // ----------------------------------------------------

    DynamicSparsityPattern dsp(l, L);
    for (unsigned int i = 0; i < l; ++i)
        for (auto dof : rows[i])
            dsp.add(i, dof);

    SparsityPattern sp_G;
    sp_G.copy_from(dsp);

    G.reinit(sp_G);
    for (unsigned int i = 0; i < l; ++i)
        for (auto dof : rows[i])
            G.set(i, dof, 1.0);

    // ----------------------------------------------------

    observation_operator_sp = utils::make_product_sparsity_AB(G, boundary_mass_matrix);
    observation_operator_matrix.reinit(observation_operator_sp);
    G.mmult(observation_operator_matrix, boundary_mass_matrix);
}


// TODO Move this with all the other products etc. into an own factory
template <int dim, typename Number>
void ObservationOperatorFactory<dim, Number>::_assemble_boundary_mass_matrix(
    const FiniteElement<dim> &fe,
    const DoFHandler<dim> &dof_handler,
    const AffineConstraints<Number> &constraints,
    const SparsityPattern &sparsity_pattern,
    SparseMatrix<Number>& boundary_mass_matrix) const
{
    QGaussLobatto<dim-1> face_quadrature_formula(2);
    FEFaceValues<dim> face_fe_values(fe, face_quadrature_formula,
                                     update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_quadrature_points = face_quadrature_formula.size();
  
    FullMatrix<Number> boundary_cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    boundary_mass_matrix.reinit(sparsity_pattern);
    boundary_mass_matrix = 0;
    
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
    for (; cell != endc; ++cell) {
        for (unsigned int face = 0; face < GeometryInfo<3>::faces_per_cell; face++) {
			if (cell->face(face)->at_boundary()) 
            {   
                boundary_cell_matrix = 0;
                cell->get_dof_indices(local_dof_indices);
                face_fe_values.reinit(cell, face);
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const unsigned int component_i = fe.system_to_component_index(i).first;

                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        const unsigned int component_j = fe.system_to_component_index(j).first;

                        if (component_i != component_j)
                            continue;

                        for (unsigned int q_point = 0; q_point < n_quadrature_points; ++q_point) {
                            boundary_cell_matrix(i, j) += 
                            face_fe_values.shape_value(i,q_point)*
                            face_fe_values.shape_value(j,q_point)*
                            face_fe_values.JxW(q_point);
                        }
                    }
                }
                constraints.distribute_local_to_global(
                    boundary_cell_matrix, 
                    local_dof_indices, 
                    boundary_mass_matrix
                );
            }
        }
    }
    constraints.condense(boundary_mass_matrix);
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