#pragma once

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>

#include "Operators.hpp"
#include "StoredEnergyFunction.hpp"

using namespace dealii;

template <int dim>
struct StoredEnergyOperatorContext {
  const FiniteElement<dim>           &fe;
  const DoFHandler<dim>              &dof_handler;
};

template <int dim, typename Number>
class StoredEnergyOperator : public BaseOperator<Number>
{
public:
    StoredEnergyOperator(
        const StoredEnergyFunction<dim, Number>& stored_energy_function, 
        const StoredEnergyOperatorContext<dim>& ctx
    );
    
    void apply(Vector<Number>       &y,
               const Vector<Number> &x) const;

private:
    const StoredEnergyFunction<dim, Number>& m_stored_energy_function;
    const StoredEnergyOperatorContext<dim>& m_ctx;
};