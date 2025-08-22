#pragma once

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/dofs/dof_handler.h>

#include "SystemMatrices.hpp"

using namespace dealii;

typedef std::variant<int, double, std::string> SystemMatrixHyperparameterType;
typedef std::map<std::string, SystemMatrixHyperparameterType>  SystemMatrixHyperparameter;

inline void check_required_double_keys(const SystemMatrixHyperparameter& params,
                                       const std::initializer_list<std::string>& required_keys) {
    for (const auto& key : required_keys) {
        auto it = params.find(key);
        if (it == params.end()) {
            throw std::runtime_error("Missing key: " + key);
        }
        if (!std::holds_alternative<double>(it->second)) {
            throw std::runtime_error("Key '" + key + "' must be a double");
        }
    }
}


enum class SystemMatrixType {
    Cosserat,
};

template <int dim, typename Number>
class MaterialMatricesFactory
{
public:
    void assemble_system_matrix(const SystemMatrixType &system_matrix_type,
                                const FiniteElement<dim> &fe,
                                const DoFHandler<dim>   &dof_handler,
                                const AffineConstraints<Number> &constraints,
                                const SparsityPattern   &sparsity_pattern,
                                const SystemMatrixHyperparameter& system_matrix_hyperparameter,
                                SystemMatrices<dim, Number> &system_matrices) const;

    void assemble_cosserat_system_matrix(const FiniteElement<dim> &fe,
                                         const DoFHandler<dim>   &dof_handler,
                                         const AffineConstraints<Number> &constraints,
                                         const SparsityPattern   &sparsity_pattern,
                                         const SystemMatrixHyperparameter& lame_coeff,
                                         SystemMatrices<dim, Number> &system_matrices) const;
};