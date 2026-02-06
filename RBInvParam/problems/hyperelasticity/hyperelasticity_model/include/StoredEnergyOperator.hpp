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

// ---------------------------------- StoredEnergyOperatorBase ----------------------------------
template <int dim, typename Number>
class StoredEnergyOperatorBase
{
protected:
  StoredEnergyOperatorBase(const Vector<Number>                    &q,
                           const Vector<Number>                    &full_q,
                           const StateSpaceContext<dim, Number>    &state_space_context,
                           const ParamSpaceContext<dim, Number>    &param_space_context,
                           const StoredEnergyFunction<dim, Number> &stored_energy_function);
  
  SparseMatrix<Number> assemble_hessian_matrix(bool param_linear_part_only = false);
  SparseMatrix<Number> assemble_hessian_matrix(
    const Vector<Number> &u, 
    bool param_linear_part_only = false
  );


  const Vector<Number>                     m_q;
  const Vector<Number>                     m_full_q;
  const StateSpaceContext<dim, Number>    &m_state_space_context;
  const ParamSpaceContext<dim, Number>    &m_param_space_context;
  const StoredEnergyFunction<dim, Number> &m_stored_energy_function;
};

// ---------------------------------- LinearStoredEnergyOperator ----------------------------------
template <int dim, typename Number>
class LinearStoredEnergyOperator : public StoredEnergyOperatorBase<dim, Number>,
                                   public SparseMatrixOperator<Number>
                                   
{
public:
  LinearStoredEnergyOperator(const Vector<Number>                    &q,
                             const Vector<Number>                    &full_q,
                             const StateSpaceContext<dim, Number>    &state_space_context,
                             const ParamSpaceContext<dim, Number>    &param_space_context,
                             const StoredEnergyFunction<dim, Number> &stored_energy_function,
                             const bool                              &param_linear_part_only = false);
         
private:
  const bool m_param_linear_part_only;
};

// ---------------------------------- StoredEnergyOperator ----------------------------------
template <int dim, typename Number>
class StoredEnergyOperator : public StoredEnergyOperatorBase<dim, Number>,
                             public BaseOperator<Number>
                             
{
public:
  StoredEnergyOperator(const Vector<Number>                    &q,
                       const Vector<Number>                    &full_q,
                       const StateSpaceContext<dim, Number>    &state_space_context,
                       const ParamSpaceContext<dim, Number>    &param_space_context,
                       const StoredEnergyFunction<dim, Number> &stored_energy_function,
                       const bool                              &param_linear_part_only = false);
  
  std::size_t dim_source() const override;
  std::size_t dim_range()  const override;
  
  void apply(Vector<Number>       &y,
             const Vector<Number> &x) const override;

  std::unique_ptr<BaseOperator<Number>> jacobian(
    const Vector<Number> &u
  ) const override;

private:
  const bool m_param_linear_part_only;
};

// ---------------------------------- StoredEnergyJacobianOperator ----------------------------------
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
                               const StoredEnergyFunction<dim, Number> &stored_energy_function,
                               const bool                              &param_linear_part_only = false);

private:
  const Vector<Number> m_u;  
  const bool m_param_linear_part_only;
};

// ---------------------------------- StoredEnergyParamDerivOperator ----------------------------------
template <int dim, typename Number>
class StoredEnergyParamDerivOperator : public StoredEnergyOperatorBase<dim, Number>,
                                       public BaseOperator<Number>
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
