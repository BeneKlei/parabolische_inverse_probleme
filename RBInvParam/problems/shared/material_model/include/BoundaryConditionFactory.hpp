#pragma once

#include <array>
#include <map>
#include <variant>
#include <iostream>

#include <deal.II/base/function.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/vector_tools.h>

#include "StateSpaceContext.hpp"

using namespace dealii;

enum class BoundaryConditionType {
  AllNeumann,
  DirichletOnYandZ
};

using BoundaryConditionHyperparameterType =
  std::variant<bool, int, double, std::string, std::vector<double>>;

using BoundaryConditionHyperparameter =
  std::map<std::string, BoundaryConditionHyperparameterType>;

template <int dim, typename Number>
class BoundaryConditionFactory
{
public:
  void assemble_constraints(const StateSpaceContext<dim, Number> &state_space_context,
                            const BoundaryConditionType &bc_type,
                            const BoundaryConditionHyperparameter &hyperparameter,
                            AffineConstraints<Number> &bc_constraints) const
  {
    bc_constraints.clear();

    switch (bc_type)
    {
      case BoundaryConditionType::AllNeumann:
      {
        std::cout << "\t\t BoundaryCondition: AllNeumann (no Dirichlet constraints)" << std::endl;
        break;
      }

      case BoundaryConditionType::DirichletOnYandZ:
      {
        std::cout << "\t\t BoundaryCondition: DirichletOnYandZ "
                  << "(boundary_ids = {3,4,5,6}, homogeneous)" << std::endl;

        assemble_dirichlet_on_yz(state_space_context, hyperparameter, bc_constraints);
        break;
      }

      default:
        AssertThrow(false, ExcMessage("Unknown BoundaryConditionType"));
    }

    bc_constraints.close();
  }

private:
  void assemble_dirichlet_on_yz(const StateSpaceContext<dim, Number> &state_space_context,
                                const BoundaryConditionHyperparameter &hyperparameter,
                                AffineConstraints<Number> &bc_constraints) const
  {
    std::array<types::boundary_id, 4> ids{3,4,5,6};

    // optional override
    auto it = hyperparameter.find("boundary_ids");
    if (it != hyperparameter.end())
    {
      const auto *v = std::get_if<std::vector<double>>(&it->second);
      AssertThrow(v != nullptr, ExcMessage("boundary_ids must be std::vector<double>."));
      AssertThrow(v->size() == 4, ExcMessage("boundary_ids must have 4 entries."));
      for (unsigned int i = 0; i < 4; ++i)
        ids[i] = static_cast<types::boundary_id>((*v)[i]);
    }

    const Functions::ZeroFunction<dim> zero(state_space_context.fe().n_components());
    for (const auto id : ids)
      VectorTools::interpolate_boundary_values(state_space_context.dof_handler(),
                                               id,
                                               zero,
                                               bc_constraints);
  }
};