#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>

#include "ObservationOperatorFactory.hpp"

template class ObservationOperatorFactory<3, double>;

template <int dim, typename Number>
void ObservationOperatorFactory<dim, Number>::assemble_observation(
    const ObservationOperatorType &observation_operator_type,
    const FiniteElement<dim> &fe,
    const DoFHandler<dim> &dof_handler,
    const AffineConstraints<Number> &constraints,
    const SparsityPattern &sparsity_pattern,
    SparseMatrix<Number>& observation_operator_matrix) const
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
        observation_operator_matrix
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
        observation_operator_matrix
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
        observation_operator_matrix
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
    SparseMatrix<Number>& observation_operator_matrix) const
{
    observation_operator_matrix.reinit(sparsity_pattern);
    observation_operator_matrix = 0;

    for (types::global_dof_index i = 0; i < dof_handler.n_dofs(); ++i)
        observation_operator_matrix.set(i, i, Number(1));
}

template <int dim, typename Number>
void ObservationOperatorFactory<dim, Number>::assemble_boundary_observation(
    const ObservationOperatorType &observation_operator_type,
    const FiniteElement<dim> &fe,
    const DoFHandler<dim> &dof_handler,
    const AffineConstraints<Number> &constraints,
    const SparsityPattern &sparsity_pattern,
    SparseMatrix<Number>& observation_operator_matrix) const
{
    ObservationOperatorFactory<dim, Number>::assemble_boundary_mass_matrix(
        fe,
        dof_handler,
        constraints,
        sparsity_pattern,
        observation_operator_matrix);
}

template <int dim, typename Number>
void ObservationOperatorFactory<dim, Number>::assemble_sensors_observation(
    const ObservationOperatorType &observation_operator_type,
    const FiniteElement<dim> &fe,
    const DoFHandler<dim> &dof_handler,
    const AffineConstraints<Number> &constraints,
    const SparsityPattern &sparsity_pattern,
    SparseMatrix<Number>& observation_operator_matrix) const
{
    // Vector<Number> G; 
    // G.reinit(dof_handler.n_dofs());
    // SensorObservationFunctions of = SensorObservationFunctions(observation_operator_type);
	// VectorTools::interpolate(dof_handler, of, G);

}


// TODO Move this with all the other products etc. into an own factory
template <int dim, typename Number>
void ObservationOperatorFactory<dim, Number>::assemble_boundary_mass_matrix(
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