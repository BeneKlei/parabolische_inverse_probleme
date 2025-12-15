#pragma once

#include <deal.II/lac/vector.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/dofs/dof_handler.h>

#include "StateProductFactory.hpp"

using namespace dealii;

enum class ObservationOperatorType {
    Identity,
    Boundary,
    Sensors,
    SensorsGrid
};


typedef std::variant<bool, int, double, std::string, std::vector<double>> ObservationOperatorHyperparameterType;
typedef std::map<std::string, ObservationOperatorHyperparameterType>  ObservationOperatorHyperparameter;

template <int dim, typename Number>
struct ObservationOperatorFactoryContext {
    const ObservationOperatorType &observation_operator_type;
    const FiniteElement<dim> &fe;
    const DoFHandler<dim> &dof_handler;
    const AffineConstraints<Number> &constraints;
    const SparsityPattern &sparsity_pattern;
    const ObservationOperatorHyperparameter &hyperparameter;
};

template <int dim, typename Number>
class ObservationOperatorFactory
{
public:
    // TODO Find a way to get a shorter wrapper for the interface
    void assemble_observation(
        const ObservationOperatorFactoryContext<dim, Number> ctx,
        SparseMatrix<Number>& observation_operator_matrix,
        SparsityPattern& observation_operator_sp
    ) const;

    void assemble_identity_observation(
        const ObservationOperatorFactoryContext<dim, Number> ctx,
        SparseMatrix<Number>& observation_operator_matrix,
        SparsityPattern& observation_operator_sp
    ) const;

    void assemble_boundary_observation(
        const ObservationOperatorFactoryContext<dim, Number> ctx,
        SparseMatrix<Number>& observation_operator_matrix,
        SparsityPattern& observation_operator_sp
    ) const;

    void assemble_sensors_observation(
        const ObservationOperatorFactoryContext<dim, Number> ctx,
        SparseMatrix<Number>& observation_operator_matrix,
        SparsityPattern& observation_operator_sp,
        std::vector<Point<dim>> sensor_points
    ) const;


    // ---------------------------- utils funcs ----------------------------
    
    std::vector<Point<dim>> _get_sensor_edges(
        const ObservationOperatorFactoryContext<dim, Number> ctx
    ) const;

    std::vector<Point<dim>> _get_sensor_grids(
        const ObservationOperatorFactoryContext<dim, Number> ctx
    ) const;

private:
    StateProductFactory<dim, Number> m_state_product_factory = StateProductFactory<3, Number>();
};