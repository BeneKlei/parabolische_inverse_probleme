#pragma once

#include <functional>
#include <utility>
#include <vector>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>

#include "Operators.hpp"
#include "MatrixOperator.hpp"
#include "StoredEnergyFunction.hpp"
#include "ParamSpaceContext.hpp"
#include "StateSpaceContext.hpp"

using namespace dealii;

// // ---- forward declarations ----
// template <int dim, typename Number>
// class StoredEnergyJacobianOperator;

// template <int dim, typename Number>
// class StoredEnergyParamDerivOperator; // optional, only if referenced early
// // ------------------------------

template <int dim, typename Number>
class StoredEnergyOperatorBase
{
protected:
  StoredEnergyOperatorBase(const Vector<Number>                    &q,
                           const Vector<Number>                    &full_q,
                           const StateSpaceContext<dim, Number>    &state_space_context,
                           const ParamSpaceContext<dim, Number>    &param_space_context,
                           const StoredEnergyFunction<dim, Number> &stored_energy_function);

  const Vector<Number>                     m_q;
  const Vector<Number>                     m_full_q;
  const StateSpaceContext<dim, Number>    &m_state_space_context;
  const ParamSpaceContext<dim, Number>    &m_param_space_context;
  const StoredEnergyFunction<dim, Number> &m_stored_energy_function;
};

template <int dim, typename Number>
class StoredEnergyOperator : public BaseOperator<Number>,
                             public StoredEnergyOperatorBase<dim, Number>
{
public:
  StoredEnergyOperator(const Vector<Number>                    &q,
                       const Vector<Number>                    &full_q,
                       const StateSpaceContext<dim, Number>    &state_space_context,
                       const ParamSpaceContext<dim, Number>    &param_space_context,
                       const StoredEnergyFunction<dim, Number> &stored_energy_function);
  
  std::size_t dim_source() const override;
  std::size_t dim_range()  const override;
  
  void apply(Vector<Number>       &y,
             const Vector<Number> &x) const override;

  std::unique_ptr<BaseOperator<Number>> jacobian(
    const Vector<Number> &u
  ) const override;
};

template <int dim, typename Number>
class StoredEnergyJacobianOperator : public StoredEnergyOperatorBase<dim, Number>,
                                     public SparseMatrixOperator<Number>
{
public:
  StoredEnergyJacobianOperator(const Vector<Number>                    &u,
                               const Vector<Number>                    &q,
                               const Vector<Number>                    &full_q,
                               const StateSpaceContext<dim, Number>    &state_space_context,
                               const ParamSpaceContext<dim, Number>    &param_space_context,
                               const StoredEnergyFunction<dim, Number> &stored_energy_function);

private:
  SparseMatrix<Number> assemble_jacobian(const Vector<Number> &u);

  const Vector<Number> m_u;
};

template <int dim, typename Number>
class StoredEnergyParamDerivOperator : public BaseOperator<Number>,
                                       public StoredEnergyOperatorBase<dim, Number>
{
public:
  StoredEnergyParamDerivOperator(const Vector<Number>                    &u,
                                 const Vector<Number>                    &q,
                                 const Vector<Number>                    &full_q,
                                 const StateSpaceContext<dim, Number>    &state_space_context,
                                 const ParamSpaceContext<dim, Number>    &param_space_context,
                                 const StoredEnergyFunction<dim, Number> &stored_energy_function);
  
  std::size_t dim_source() const override;
  std::size_t dim_range()  const override;

  void apply(Vector<Number>       &y,
             const Vector<Number> &d) const override;

  void apply_adjoint(Vector<Number>       &y,
                     const Vector<Number> &p) const override;

private:  
  const Vector<Number> m_u;
};
