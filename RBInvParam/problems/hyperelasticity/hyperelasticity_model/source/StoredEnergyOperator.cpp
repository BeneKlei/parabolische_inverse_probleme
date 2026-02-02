#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include "StoredEnergyOperator.hpp"

using namespace dealii;

// ---------------------------------- StoredEnergyOperatorBase ----------------------------------
template <int dim, typename Number>
StoredEnergyOperatorBase<dim, Number>::StoredEnergyOperatorBase(
    const Vector<Number>                    &q,
    const Vector<Number>                    &full_q,
    const StateSpaceContext<dim, Number>    &state_space_context,
    const ParamSpaceContext<dim, Number>    &param_space_context,
    const StoredEnergyFunction<dim, Number> &stored_energy_function)
  : m_q(q)
  , m_full_q(full_q)
  , m_state_space_context(state_space_context)
  , m_param_space_context(param_space_context)
  , m_stored_energy_function(stored_energy_function)  
{}

// ---------------------------------- StoredEnergyOperator ----------------------------------
template <int dim, typename Number>
StoredEnergyOperator<dim, Number>::StoredEnergyOperator(
    const Vector<Number>                    &q,
    const Vector<Number>                    &full_q,
    const StateSpaceContext<dim, Number>    &state_space_context,
    const ParamSpaceContext<dim, Number>    &param_space_context,
    const StoredEnergyFunction<dim, Number> &stored_energy_function)
  : BaseOperator<Number>(stored_energy_function.m_linear)
  , StoredEnergyOperatorBase<dim, Number>(q,
                                          full_q,
                                          state_space_context,
                                          param_space_context,
                                          stored_energy_function)
{}
template <int dim, typename Number>
std::size_t StoredEnergyOperator<dim, Number>::dim_source() const
{
  return this->m_state_space_context.dof_handler().n_dofs();
}

template <int dim, typename Number>
std::size_t StoredEnergyOperator<dim, Number>::dim_range() const
{
  return this->m_state_space_context.dof_handler().n_dofs();
}

template <int dim, typename Number>
void StoredEnergyOperator<dim, Number>::apply(Vector<Number>       &y,
                                              const Vector<Number> &u) const
{
  AssertDimension(u.size(), dim_source());
  y.reinit(dim_range());
  y = Number(0);

  const unsigned int n_q = this->m_state_space_context.quadrature().size();
  FEValues<dim> fe_values_state(this->m_state_space_context.fe(),
                                this->m_state_space_context.quadrature(),
                                update_gradients | update_JxW_values |
                                update_quadrature_points | update_values);

  const unsigned int dofs_per_cell = this->m_state_space_context.fe().dofs_per_cell;

  // --- local storage ---
  Vector<Number> local_y(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const FEValuesExtractors::Vector vel(0);

  // TODO Multi query scenario. Alloc once and reuse.
  std::vector<Tensor<2, dim>> u_gradients(n_q);
  std::vector<Number>         param_values(n_q, Number(1.0));

  const Tensor<2, dim> I(unit_symmetric_tensor<dim, Number>());

  for (const auto &cell : this->m_state_space_context.dof_handler().active_cell_iterators())
  {
    local_y = Number(0);

    fe_values_state.reinit(cell);
    fe_values_state[vel].get_function_gradients(u, u_gradients);
    const auto &q_points = fe_values_state.get_quadrature_points();

    if (this->m_q.size() == 0)
      throw std::runtime_error("StoredEnergyOperator: q is required but is empty.");
      
    this->m_param_space_context.evaluate_values(this->m_q, q_points, param_values);

    for (unsigned int q = 0; q < n_q; ++q)
    {      
      const auto &q_point        = q_points[q];
      const auto &u_grad         = u_gradients[q];
      const Number &param_value  = param_values[q];

      const Tensor<2, dim> DY = this->m_stored_energy_function.gradient(q_point, u_grad + I);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const unsigned int component_i =
            this->m_state_space_context.fe().system_to_component_index(i).first;
        
        local_y(i) += param_value * DY[component_i] *
                      fe_values_state.shape_grad(i, q) *
                      fe_values_state.JxW(q);

        // local_y(i) += param_value *
        //       scalar_product(DY, fe_values_state[vel].gradient(i,q)) *
        //       fe_values_state.JxW(q);
      }
    }

    cell->get_dof_indices(local_dof_indices);
    this->m_state_space_context.BC_constraints().distribute_local_to_global(
      local_y,
      local_dof_indices,
      y
    );

    // for (unsigned int i = 0; i < dofs_per_cell; ++i)
    //   y(local_dof_indices[i]) += local_y(i);
  }
  y.compress(VectorOperation::add);
}

template <int dim, typename Number>
std::unique_ptr<BaseOperator<Number>>
StoredEnergyOperator<dim, Number>::jacobian(const Vector<Number> &u) const
{
  return std::make_unique<StoredEnergyJacobianOperator<dim, Number>>(
        u,
        this->m_q,
        this->m_full_q,
        this->m_state_space_context,
        this->m_param_space_context,
        this->m_stored_energy_function
    );
}

// ---------------------------------- StoredEnergyJacobianOperator ----------------------------------
template <int dim, typename Number>
StoredEnergyJacobianOperator<dim, Number>::StoredEnergyJacobianOperator(
    const Vector<Number>                    &u,
    const Vector<Number>                    &q,
    const Vector<Number>                    &full_q,
    const StateSpaceContext<dim, Number>    &state_space_context,
    const ParamSpaceContext<dim, Number>    &param_space_context,
    const StoredEnergyFunction<dim, Number> &stored_energy_function)
  : StoredEnergyOperatorBase<dim, Number>(q,
                                          full_q,
                                          state_space_context,
                                          param_space_context,
                                          stored_energy_function)
  , SparseMatrixOperator<Number>(assemble_jacobian(u))
  , m_u(u)                                          
{}

template <int dim, typename Number>
SparseMatrix<Number> StoredEnergyJacobianOperator<dim, Number>::assemble_jacobian(const Vector<Number> &u)
{
  const unsigned int n_q = this->m_state_space_context.quadrature().size();
  // --- FEValues for state ---
  FEValues<dim> fe_values_state(this->m_state_space_context.fe(),
                                this->m_state_space_context.quadrature(),
                                update_gradients | update_JxW_values |
                                update_quadrature_points | update_values);
  
  const unsigned int dofs_per_cell = this->m_state_space_context.fe().dofs_per_cell;

  // --- local storage ---
  FullMatrix<Number> local_J(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const FEValuesExtractors::Vector vel(0);
  Tensor<2,dim> test_j_H;
  
  std::vector<Tensor<2, dim>> u_gradients(n_q);
  std::vector<Number>         param_values(n_q, Number(1.0));

  const Tensor<2, dim> I(unit_symmetric_tensor<dim, Number>());

  SparseMatrix<Number> J;
  J.reinit(this->m_state_space_context.state_sp());
  J = Number(0);

  for (const auto &cell : this->m_state_space_context.dof_handler().active_cell_iterators())
  {
    local_J = Number(0);

    fe_values_state.reinit(cell);
    fe_values_state[vel].get_function_gradients(u, u_gradients);
    const auto &q_points = fe_values_state.get_quadrature_points();

    if (this->m_q.size() == 0)
      throw std::runtime_error("StoredEnergyJacobianOperator: q is required but is empty.");
    this->m_param_space_context.evaluate_values(this->m_q, q_points, param_values);

    for (unsigned int q_point_id = 0; q_point_id < n_q; ++q_point_id)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const auto &q_point        = q_points[q_point_id];
        const auto &u_grad         = u_gradients[q_point_id];
        const Number &param_value  = param_values[q_point_id];

        const unsigned int component_i = 
          this->m_state_space_context.fe().system_to_component_index(i).first;
        

        // TODO Refactor avoiding the "trick" with test_j_H
        const Tensor<2, dim> DYDYH =
          this->m_stored_energy_function.contracted_hessian(
            q_point, 
            u_grad + I,
            fe_values_state.shape_grad(i, q_point_id),
            component_i
        );

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          const unsigned int component_j = 
            this->m_state_space_context.fe().system_to_component_index(j).first;
          
          test_j_H.clear();
          test_j_H[component_j] = fe_values_state.shape_grad(j, q_point_id);

          // TODO Refactor using deal.ii Tensor methods
          local_J(i, j) += param_value * 
                           double_contract<0,0,1,1>(DYDYH, test_j_H) * 
                           fe_values_state.JxW(q_point_id);
        }
      }
    }

    cell->get_dof_indices(local_dof_indices); 
    this->m_state_space_context.BC_constraints().distribute_local_to_global(
      local_J,
      local_dof_indices,
      J
    );
  }

  J.compress(VectorOperation::add);
  return J;
}

// ---------------------------------- StoredEnergyParamDerivOperator ----------------------------------// ---------------------------------- StoredEnergyParamDerivOperator ----------------------------------

template <int dim, typename Number>
StoredEnergyParamDerivOperator<dim, Number>::StoredEnergyParamDerivOperator(
    const Vector<Number>                    &u,
    const Vector<Number>                    &q,
    const Vector<Number>                    &full_q,
    const StateSpaceContext<dim, Number>    &state_space_context,
    const ParamSpaceContext<dim, Number>    &param_space_context,
    const StoredEnergyFunction<dim, Number> &stored_energy_function)
  : BaseOperator<Number>(true)
  , StoredEnergyOperatorBase<dim, Number>(q,
                                          full_q,
                                          state_space_context,
                                          param_space_context,
                                          stored_energy_function)
  , m_u(u)
{}

template <int dim, typename Number>
std::size_t StoredEnergyParamDerivOperator<dim, Number>::dim_source() const
{
  return this->m_param_space_context.free_dofs().size();
}

template <int dim, typename Number>
std::size_t StoredEnergyParamDerivOperator<dim, Number>::dim_range() const
{
  return this->m_state_space_context.dof_handler().n_dofs();
}

template <int dim, typename Number>
void StoredEnergyParamDerivOperator<dim, Number>::apply(Vector<Number>       &y,
                                                        const Vector<Number> &d) const
{
  AssertDimension(d.size(), dim_source());
  y.reinit(this->dim_range());
  y = Number(0);

  const unsigned int n_q = this->m_state_space_context.quadrature().size();
  FEValues<dim> fe_values_state(this->m_state_space_context.fe(),
                                this->m_state_space_context.quadrature(),
                                update_gradients | update_JxW_values |
                                update_quadrature_points | update_values);

  const unsigned int dofs_per_cell = this->m_state_space_context.fe().dofs_per_cell;

  // --- local storage ---
  Vector<Number> local_y(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const FEValuesExtractors::Vector vel(0);

  // TODO Multi query scenario. Alloc once and reuse.
  std::vector<Tensor<2, dim>> u_gradients(n_q);
  std::vector<Number>         param_values(n_q);

  const Tensor<2, dim> I(unit_symmetric_tensor<dim, Number>());

  for (const auto &cell : this->m_state_space_context.dof_handler().active_cell_iterators())
  {
    local_y = Number(0);

    fe_values_state.reinit(cell);
    fe_values_state[vel].get_function_gradients(this->m_u, u_gradients);
    const auto &q_points = fe_values_state.get_quadrature_points();

    // TODO Cache them once, reuse for all u's and the jacobian
    this->m_param_space_context.evaluate_values(
      d,
      q_points,
      param_values
    );

    for (unsigned int q = 0; q < n_q; ++q)
    {      
      const auto &q_point        = q_points[q];
      const auto &u_grad         = u_gradients[q];
      const Number &param_value  = param_values[q];

      const Tensor<2, dim> DY = this->m_stored_energy_function.gradient(q_point, u_grad + I);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const unsigned int component_i =
            this->m_state_space_context.fe().system_to_component_index(i).first;
        
        local_y(i) += param_value * DY[component_i] *
                      fe_values_state.shape_grad(i, q) *
                      fe_values_state.JxW(q);

        // local_y(i) += param_value *
        //       scalar_product(DY, fe_values_state[vel].gradient(i,q)) *
        //       fe_values_state.JxW(q);
      }
    }

    cell->get_dof_indices(local_dof_indices);
    this->m_state_space_context.BC_constraints().distribute_local_to_global(
      local_y,
      local_dof_indices,
      y
    );
  }
  y.compress(VectorOperation::add);
}


template <int dim, typename Number>
void StoredEnergyParamDerivOperator<dim, Number>::apply_adjoint(Vector<Number>       &y,
                                                                const Vector<Number> &p) const
{
  AssertDimension(d.size(), dim_range());
  y.reinit(this->dim_source());
  y = Number(0);

  const unsigned int n_q = this->m_state_space_context.quadrature().size();
  FEValues<dim> fe_values_state(this->m_state_space_context.fe(),
                                this->m_state_space_context.quadrature(),
                                update_gradients | update_JxW_values |
                                update_quadrature_points | update_values);

  const unsigned int dofs_per_cell = this->m_state_space_context.fe().dofs_per_cell;

  // --- local storage ---
  //Vector<Number> local_y(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const FEValuesExtractors::Vector vel(0);

  // TODO Multi query scenario. Alloc once and reuse.
  std::vector<Tensor<2, dim>> u_gradients(n_q);
  std::vector<Tensor<2, dim>> p_gradients(n_q);
  std::vector<Number>         param_values(n_q);
  Vector<Number>              e(this->dim_source());

  const Tensor<2, dim> I(unit_symmetric_tensor<dim, Number>());

  for (unsigned int i = 0; i < this->dim_source(); ++i)
  {
    e = Number(0.0);
    e(i) = Number(1.0);

    for (const auto &cell : this->m_state_space_context.dof_handler().active_cell_iterators())
    {
      fe_values_state.reinit(cell);
      fe_values_state[vel].get_function_gradients(this->m_u, u_gradients);
      fe_values_state[vel].get_function_gradients(p, p_gradients);
      const auto &q_points = fe_values_state.get_quadrature_points();

      this->m_param_space_context.evaluate_values(
        e,
        q_points,
        param_values,
        true
      );

      for (unsigned int q = 0; q < n_q; ++q)
      {
        const auto &q_point        = q_points[q];
        const auto &u_grad         = u_gradients[q];
        const auto &p_grad         = p_gradients[q];
        const Number &param_value  = param_values[q];

        const Tensor<2, dim> DY = this->m_stored_energy_function.gradient(q_point, u_grad + I);      
        const Number DYp_grad = scalar_product(DY, p_grad); 

        y(i) += param_value * DYp_grad * fe_values_state.JxW(q);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------- 
template class StoredEnergyOperator<2, double>;
template class StoredEnergyOperator<3, double>;
template class StoredEnergyJacobianOperator<2, double>;
template class StoredEnergyJacobianOperator<3, double>;
template class StoredEnergyParamDerivOperator<2, double>;
template class StoredEnergyParamDerivOperator<3, double>;
