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

template <int dim, typename Number>
SparseMatrix<Number> StoredEnergyOperatorBase<dim, Number>::assemble_hessian_matrix(
  bool param_linear_part_only
)
{
  AssertThrow(
    m_stored_energy_function.m_linear, 
    ExcMessage("Stored energy function is not linear. Hessian depends on state u.")
  );

   Vector<Number> _u;
   _u.reinit(m_state_space_context.dof_handler().n_dofs());
   _u = Number(0.0);
   return assemble_hessian_matrix(_u, param_linear_part_only);
}

template <int dim, typename Number>
SparseMatrix<Number> StoredEnergyOperatorBase<dim, Number>::assemble_hessian_matrix(
  const Vector<Number> &u,
  bool param_linear_part_only
)
{
  const unsigned int n_q = m_state_space_context.quadrature().size();
  // --- FEValues for state ---
  FEValues<dim> fe_values_state(m_state_space_context.mapping(),
                                m_state_space_context.fe(),
                                m_state_space_context.quadrature(),
                                update_gradients | update_JxW_values |
                                update_quadrature_points | update_values);
  
  const unsigned int dofs_per_cell = m_state_space_context.fe().dofs_per_cell;

  // --- local storage ---
  FullMatrix<Number> local_J(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const FEValuesExtractors::Vector vel(0);
  Tensor<2,dim> test_j_H;
  
  std::vector<Tensor<2, dim>> u_gradients(n_q);
  std::vector<Number>         param_values(n_q, Number(1.0));

  const Tensor<2, dim> I(unit_symmetric_tensor<dim, Number>());

  SparseMatrix<Number> J;
  J.reinit(m_state_space_context.sparsity_pattern());
  J = Number(0);

  Vector<Number> q_grid;
  this->m_param_space_context.free_to_grid(m_q, q_grid, param_linear_part_only);

  for (const auto &cell : m_state_space_context.dof_handler().active_cell_iterators())
  {
    local_J = Number(0);

    fe_values_state.reinit(cell);
    fe_values_state[vel].get_function_gradients(u, u_gradients);
    const auto &q_points = fe_values_state.get_quadrature_points();

    if (m_q.size() == 0)
      throw std::runtime_error("StoredEnergyJacobianOperator: q is required but is empty.");
    
    m_param_space_context.evaluate_values(
      q_grid, 
      q_points, 
      param_values,
      param_linear_part_only
    );

    for (unsigned int q_point_id = 0; q_point_id < n_q; ++q_point_id)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const auto &q_point        = q_points[q_point_id];
        const auto &u_grad         = u_gradients[q_point_id];
        const Number &param_value  = param_values[q_point_id];

        if (param_value == Number(0.0))
          continue;

        const unsigned int component_i = 
          m_state_space_context.fe().system_to_component_index(i).first;
        

        // TODO Refactor avoiding the "trick" with test_j_H
        const Tensor<2, dim> DYDYH =
          m_stored_energy_function.contracted_hessian(
            q_point, 
            u_grad + I,
            fe_values_state.shape_grad(i, q_point_id),
            component_i
        );

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          const unsigned int component_j = 
            m_state_space_context.fe().system_to_component_index(j).first;
          
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
    m_state_space_context.boundary_constraints().distribute_local_to_global(
      local_J,
      local_dof_indices,
      J
    );
  }

  J.compress(VectorOperation::add);
  return J;
}

// ---------------------------------- LinearStoredEnergyOperator ----------------------------------
template <int dim, typename Number>
LinearStoredEnergyOperator<dim, Number>::LinearStoredEnergyOperator(
    const Vector<Number>                    &q,
    const Vector<Number>                    &full_q,
    const StateSpaceContext<dim, Number>    &state_space_context,
    const ParamSpaceContext<dim, Number>    &param_space_context,
    const StoredEnergyFunction<dim, Number> &stored_energy_function,
    const bool                              &param_linear_part_only)
  : StoredEnergyOperatorBase<dim, Number>(q,
                                          full_q,
                                          state_space_context,
                                          param_space_context,
                                          stored_energy_function)
  , SparseMatrixOperator<Number>(this->assemble_hessian_matrix(param_linear_part_only))
  , m_param_linear_part_only(param_linear_part_only)
{
  Assert(this->m_stored_energy_function.m_linear, ExcMessage(
      "StoredEnergyFunction must be linear for LinearStoredEnergyOperator."));
}

// ---------------------------------- StoredEnergyOperator ----------------------------------
template <int dim, typename Number>
StoredEnergyOperator<dim, Number>::StoredEnergyOperator(
    const Vector<Number>                    &q,
    const Vector<Number>                    &full_q,
    const StateSpaceContext<dim, Number>    &state_space_context,
    const ParamSpaceContext<dim, Number>    &param_space_context,
    const StoredEnergyFunction<dim, Number> &stored_energy_function,
    const bool                              &param_linear_part_only) 
  : StoredEnergyOperatorBase<dim, Number>(q,
                                          full_q,
                                          state_space_context,
                                          param_space_context,
                                          stored_energy_function)
  , BaseOperator<Number>(stored_energy_function.m_linear)
  , m_param_linear_part_only(param_linear_part_only)
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
  FEValues<dim> fe_values_state(this->m_state_space_context.mapping(),
                                this->m_state_space_context.fe(),
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

  Vector<Number> q_grid;
  this->m_param_space_context.free_to_grid(this->m_q, q_grid, this->m_param_linear_part_only);

  for (const auto &cell : this->m_state_space_context.dof_handler().active_cell_iterators())
  {
    local_y = Number(0);

    fe_values_state.reinit(cell);
    fe_values_state[vel].get_function_gradients(u, u_gradients);
    const auto &q_points = fe_values_state.get_quadrature_points();

    if (this->m_q.size() == 0)
      throw std::runtime_error("StoredEnergyOperator: q is required but is empty.");
      
    this->m_param_space_context.evaluate_values(
      q_grid, 
      q_points, 
      param_values,
      this->m_param_linear_part_only
    );

    for (unsigned int q = 0; q < n_q; ++q)
    {      
      const auto &q_point        = q_points[q];
      const auto &u_grad         = u_gradients[q];
      const Number &param_value  = param_values[q];

      if (param_value == Number(0.0))
        continue;

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
    this->m_state_space_context.boundary_constraints().distribute_local_to_global(
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
        this->m_stored_energy_function,
        this->m_param_linear_part_only
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
    const StoredEnergyFunction<dim, Number> &stored_energy_function,
    const bool                              &param_linear_part_only)
  : StoredEnergyOperatorBase<dim, Number>(q,
                                          full_q,
                                          state_space_context,
                                          param_space_context,
                                          stored_energy_function)
  , SparseMatrixOperator<Number>(this->assemble_hessian_matrix(u, param_linear_part_only))
  , m_u(u)
  , m_param_linear_part_only(param_linear_part_only)                                         
{}

// ---------------------------------- StoredEnergyParamDerivOperator ----------------------------------

template <int dim, typename Number>
StoredEnergyParamDerivOperator<dim, Number>::StoredEnergyParamDerivOperator(
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
  , BaseOperator<Number>(true)
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
  FEValues<dim> fe_values_state(this->m_state_space_context.mapping(),
                                this->m_state_space_context.fe(),
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

  Vector<Number> d_grid;
  this->m_param_space_context.free_to_grid(d, d_grid, /*linear_part=*/true);

  for (const auto &cell : this->m_state_space_context.dof_handler().active_cell_iterators())
  {
    local_y = Number(0);

    fe_values_state.reinit(cell);
    fe_values_state[vel].get_function_gradients(this->m_u, u_gradients);
    const auto &q_points = fe_values_state.get_quadrature_points();

    // TODO Cache them once, reuse for all u's and the jacobian
    this->m_param_space_context.evaluate_values(
      d_grid,
      q_points,
      param_values,
      true
    );

    for (unsigned int q = 0; q < n_q; ++q)
    {      
      const auto &q_point        = q_points[q];
      const auto &u_grad         = u_gradients[q];
      const Number &param_value  = param_values[q];

      if (param_value == Number(0.0))
        continue;

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
    this->m_state_space_context.boundary_constraints().distribute_local_to_global(
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
  AssertDimension(p.size(), dim_range());
  y.reinit(this->dim_source());
  y = Number(0);

  const unsigned int n_q = this->m_state_space_context.quadrature().size();

  FEValues<dim> fe_values_state(this->m_state_space_context.mapping(),
                                this->m_state_space_context.fe(),
                                this->m_state_space_context.quadrature(),
                                update_gradients | update_JxW_values |
                                update_quadrature_points | update_values);

  FEValues<dim> fe_values_param(this->m_param_space_context.mapping(),
                                this->m_param_space_context.fe(),
                                this->m_param_space_context.quadrature(),
                                update_gradients | update_JxW_values |
                                update_quadrature_points | update_values);

  const unsigned int dofs_per_cell_state = this->m_state_space_context.fe().dofs_per_cell;
  const unsigned int dofs_per_cell_param = this->m_param_space_context.fe().dofs_per_cell;

  // --- local storage ---
  Vector<Number> local_y(dofs_per_cell_param);
  std::vector<types::global_dof_index> local_dof_indices_param(dofs_per_cell_param);
  
  const FEValuesExtractors::Vector vel(0);

  // TODO Multi query scenario. Alloc once and reuse.
  std::vector<Tensor<2, dim>> u_gradients(n_q);
  std::vector<Tensor<2, dim>> p_gradients(n_q);

  const Tensor<2, dim> I(unit_symmetric_tensor<dim, Number>());
  
  const auto &dh_state = this->m_state_space_context.dof_handler();
  const auto &dh_param = this->m_param_space_context.dof_handler();
  const auto &tria     = dh_state.get_triangulation();

  Vector<Number> y_full;
  y_full.reinit(dh_param.n_dofs());
  y_full = Number(0);

  for (const auto &cell : tria.active_cell_iterators())
  {
    local_y = Number(0);

    const auto cell_state = typename DoFHandler<dim>::active_cell_iterator(&tria,
                          cell->level(), cell->index(), &dh_state);
    const auto cell_param = typename DoFHandler<dim>::active_cell_iterator(&tria,
                          cell->level(), cell->index(), &dh_param);

    fe_values_state.reinit(cell_state);
    fe_values_param.reinit(cell_param);

    fe_values_state[vel].get_function_gradients(this->m_u, u_gradients);
    fe_values_state[vel].get_function_gradients(p,         p_gradients);

    const auto &q_points = fe_values_state.get_quadrature_points();
    
    for (unsigned int q = 0; q < n_q; ++q)
    {
      const auto &q_point        = q_points[q];
      const auto &u_grad         = u_gradients[q];
      const auto &p_grad         = p_gradients[q];

      const Tensor<2, dim> DY = this->m_stored_energy_function.gradient(q_point, u_grad + I);
      const Number w = scalar_product(DY, p_grad) * fe_values_state.JxW(q);

      if (w == Number(0))
        continue;

      for (unsigned int i = 0; i < dofs_per_cell_param; ++i)
      {
        const Number phi_i = fe_values_param.shape_value(i, q);
        local_y(i) += phi_i * w;
      }
    }

    cell_param->get_dof_indices(local_dof_indices_param);
    this->m_param_space_context.constraints().distribute_local_to_global(
      local_y,
      local_dof_indices_param,
      y_full
    );
  }

  y_full.compress(VectorOperation::add);
  //this->m_param_space_context.constraints().set_zero(y_full);
  this->m_param_space_context.project_to_free_param(y_full, y);
}

template <int dim, typename Number>
void StoredEnergyParamDerivOperator<dim, Number>::apply_adjoint(Vector<Number>       &y,
                                                                const Vector<Number> &p) const
{
  AssertDimension(d.size(), dim_range());
  y.reinit(this->dim_source());
  y = Number(0);

  const unsigned int n_q = this->m_state_space_context.quadrature().size();
  FEValues<dim> fe_values_state(this->m_state_space_context.mapping(),
                                this->m_state_space_context.fe(),
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

  for (const auto &cell : this->m_state_space_context.dof_handler().active_cell_iterators())
  {
    fe_values_state.reinit(cell);
    fe_values_state[vel].get_function_gradients(this->m_u, u_gradients);
    fe_values_state[vel].get_function_gradients(p, p_gradients);
    const auto &q_points = fe_values_state.get_quadrature_points();
    
    for (unsigned int q = 0; q < n_q; ++q)
    {
      const auto &q_point        = q_points[q];
      const auto &u_grad         = u_gradients[q];
      const auto &p_grad         = p_gradients[q];

      Number param_value = Number(0.0);

      const Tensor<2, dim> DY = this->m_stored_energy_function.gradient(q_point, u_grad + I);      
      const Number DYp_grad = scalar_product(DY, p_grad); 

      for (unsigned int i = 0; i < this->dim_source(); ++i)
      {
        e = Number(0.0);
        e(i) = Number(1.0);

        param_value = this->m_param_space_context.evaluate_value(
          e,
          q_point,
          true
        );

        y(i) += param_value * DYp_grad * fe_values_state.JxW(q);
      }
    }
  }
}


// ---------------------------------------------------------------------------------------------------- 
template class LinearStoredEnergyOperator<2, double>;
template class LinearStoredEnergyOperator<3, double>;
template class StoredEnergyOperator<2, double>;
template class StoredEnergyOperator<3, double>;
template class StoredEnergyJacobianOperator<2, double>;
template class StoredEnergyJacobianOperator<3, double>;
template class StoredEnergyParamDerivOperator<2, double>;
template class StoredEnergyParamDerivOperator<3, double>;
