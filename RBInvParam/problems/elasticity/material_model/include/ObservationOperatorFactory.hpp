#pragma once

#include <deal.II/lac/vector.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/dofs/dof_handler.h>

using namespace dealii;

enum class ObservationOperatorType {
    Identity,
    Boundary,
    SensorsR9d,
    SensorsR8d,
};

template <int dim, typename Number>
class ObservationOperatorFactory
{
public:
    // TODO Find a way to get a shorter wrapper for the interface
    void assemble_observation(
        const ObservationOperatorType &observation_operator_type,
        const FiniteElement<dim> &fe,
        const DoFHandler<dim> &dof_handler,
        const AffineConstraints<Number> &constraints,
        const SparsityPattern &sparsity_pattern,
        SparseMatrix<Number>& observation_operator_matrix
    ) const;

    void assemble_identity_observation(
        const ObservationOperatorType &observation_operator_type,
        const FiniteElement<dim> &fe,
        const DoFHandler<dim> &dof_handler,
        const AffineConstraints<Number> &constraints,
        const SparsityPattern &sparsity_pattern,
        SparseMatrix<Number>& observation_operator_matrix
    ) const;

    void assemble_boundary_observation(
        const ObservationOperatorType &observation_operator_type,
        const FiniteElement<dim> &fe,
        const DoFHandler<dim> &dof_handler,
        const AffineConstraints<Number> &constraints,
        const SparsityPattern &sparsity_pattern,
        SparseMatrix<Number>& observation_operator_matrix
    ) const;

    void assemble_sensors_observation(
        const ObservationOperatorType &observation_operator_type,
        const FiniteElement<dim> &fe,
        const DoFHandler<dim> &dof_handler,
        const AffineConstraints<Number> &constraints,
        const SparsityPattern &sparsity_pattern,
        SparseMatrix<Number>& observation_operator_matrix
    ) const;

    void assemble_boundary_mass_matrix(
        const FiniteElement<dim> &fe,
        const DoFHandler<dim> &dof_handler,
        const AffineConstraints<Number> &constraints,
        const SparsityPattern &sparsity_pattern,
        SparseMatrix<Number>& boundary_mass_matrix
    ) const;


};