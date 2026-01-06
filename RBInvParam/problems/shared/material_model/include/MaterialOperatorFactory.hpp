#pragma once

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/dofs/dof_handler.h>


using namespace dealii;

typedef std::variant<int, double, std::string> SystemMatrixHyperparameterType;
typedef std::map<std::string, SystemMatrixHyperparameterType>  SystemOperatorHyperparameter;

template <typename T>
constexpr const char* type_name() {
    if constexpr (std::is_same_v<T, double>) return "double";
    else if constexpr (std::is_same_v<T, std::string>) return "string";
    else if constexpr (std::is_same_v<T, int>) return "int";
    else return "unknown";
}

template <class T>
inline void check_required_keys(
    const SystemOperatorHyperparameter& params,
    const std::initializer_list<std::string>& required_keys) 
{
    for (const auto& key : required_keys) {
        auto it = params.find(key);
        if (it == params.end()) {
            throw std::runtime_error("Missing key: " + key);
        }
        if (!std::holds_alternative<T>(it->second)) {
            throw std::runtime_error(
                "Key '" + key + "' must be of type " + std::string(type_name<T>())
            );
        }
    }
}

enum class MaterialOperatorType {
    CosseratSpatial,
    CosseratDelamination,
    Cosserat
};

template <int dim, typename Number>
struct MaterialOperatorFactoryContext {
  const MaterialOperatorType         &system_operator_type;
  const FiniteElement<dim>           &fe;
  const DoFHandler<dim>              &dof_handler;
  const AffineConstraints<Number>    &BC_constraints;
  const SparsityPattern              &sparsity_pattern;
  const SystemOperatorHyperparameter   &hyperparameter;
};

template <int dim, typename Number>
class MaterialOperatorFactory
{
public:
  using SparseMatrix = dealii::SparseMatrix<double>;
  // New-only API (what you want to call)
  void assemble_system(const MaterialOperatorFactoryContext<dim, Number>& ctx,
                       std::vector<SparseMatrix>& matrices,
                       bool& affine) const;

  void assemble_cosserat_system(const MaterialOperatorFactoryContext<dim, Number>& ctx,
                                std::vector<SparseMatrix>& matrices,
                                bool& affine) const;

  void assemble_cosserat_spatial_system(const MaterialOperatorFactoryContext<dim, Number>& ctx,
                                        std::vector<SparseMatrix>& matrices,
                                        bool& affine) const;

  void assemble_cosserat_delamination_system(const MaterialOperatorFactoryContext<dim, Number>& ctx,
                                             std::vector<SparseMatrix>& matrices,
                                             bool& affine) const;
};