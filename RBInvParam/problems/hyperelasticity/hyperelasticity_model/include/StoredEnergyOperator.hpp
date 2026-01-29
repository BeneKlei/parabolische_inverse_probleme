#pragma once

#include <functional>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>

#include "Operators.hpp"
#include "MatrixOperator.hpp"
#include "StoredEnergyFunction.hpp"
#include "ParamSpaceContext.hpp"
#include "StateSpaceContext.hpp"

using namespace dealii;

template <int dim, typename Number>
class StoredEnergyOperator : public BaseOperator<Number>
{
public:
  StoredEnergyOperator(const Vector<Number>                    &q,
                       const Vector<Number>                    &full_q,
                       const StateSpaceContext<dim, Number>    &state_space_context,
                       const ParamSpaceContext<dim, Number>    &param_space_context,
                       const StoredEnergyFunction<dim, Number> &stored_energy_function);

  void apply(Vector<Number>       &y,
             const Vector<Number> &x) const override;
  
  std::unique_ptr<BaseOperator<Number>> jacobian(const Vector<Number> &u) const override;


  std::size_t dim_source() const override;
  std::size_t dim_range()  const override;

private:
  const Vector<Number>                     m_q;
  const Vector<Number>                     m_full_q;

  std::vector<Number>                      m_param_values;

  const StateSpaceContext<dim, Number>    &m_state_space_context;
  const ParamSpaceContext<dim, Number>    &m_param_space_context;

  const StoredEnergyFunction<dim, Number> &m_stored_energy_function;


};
