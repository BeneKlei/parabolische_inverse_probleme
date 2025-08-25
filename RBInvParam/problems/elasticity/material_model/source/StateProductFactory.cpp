#include <deal.II/numerics/vector_tools.h>
#include <deal.II/dofs/dof_tools.h>

#include "StateProductFactory.hpp"

template class StateProductFactory<3, double>;

template <int dim, typename Number>
void StateProductFactory<dim, Number>::assemble_state_product(
  const StateProductFactoryContext<dim, Number>& ctx,
  SparseMatrix<Number>& state_product_matrix) const
{
  switch (ctx.state_product_type)
  {
  case StateProductType::L2:
    std::cout << "\t Using L2 StateProduct" << std::endl;
    StateProductFactory::assemble_l2_product(
        ctx,
        state_product_matrix
    );
    break;
  case StateProductType::L2_0:
    std::cout << "\t Using L2_0 StateProduct" << std::endl;
    StateProductFactory::assemble_l2_0_product(
        ctx,
        state_product_matrix
    );
    break;
  case StateProductType::H1_semi:
    std::cout << "\t Using H1_semi StateProduct" << std::endl;
    StateProductFactory::assemble_h1_semi_product(
        ctx,
        state_product_matrix
    );
    break;
  case StateProductType::H1_0_semi:
    std::cout << "\t Using H1_0_semi StateProduct" << std::endl;
    StateProductFactory::assemble_h1_0_semi_product(
        ctx,
        state_product_matrix
    );
    break;
  case StateProductType::H1:
    std::cout << "\t Using H1 StateProduct" << std::endl;
    StateProductFactory::assemble_h1_product(
        ctx,
        state_product_matrix
    );
    break;
  case StateProductType::H1_0:
    std::cout << "\t Using H1_0 StateProduct" << std::endl;
    StateProductFactory::assemble_h1_0_product(
        ctx,
        state_product_matrix
    );
    break;
  case StateProductType::Mass:
    std::cout << "\t Using Mass StateProduct" << std::endl;
    StateProductFactory::assemble_mass_product(
        ctx,
        state_product_matrix
    );
    break;
  case StateProductType::BoundaryMass:
    std::cout << "\t Using BoundaryMass StateProduct" << std::endl;
    StateProductFactory::assemble_boundary_mass_product(
        ctx,
        state_product_matrix
    );
    break;

  default:
    throw std::runtime_error("Unknown StateProduct type.");
  }
}


template <int dim, typename Number>
void StateProductFactory<dim, Number>::assemble_l2_product(
  const StateProductFactoryContext<dim, Number>& ctx,
  SparseMatrix<Number>& state_product_matrix) const
{
    auto l2_integrand =
    [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
        return fe.shape_value(i, q) * fe.shape_value(j, q);
    };
    AffineConstraints<Number> empty_BC_constraints;
    empty_BC_constraints.clear(); 
    empty_BC_constraints.close(); 
    this->_assemble_product(ctx, state_product_matrix, l2_integrand, empty_BC_constraints);
}

template <int dim, typename Number>
void StateProductFactory<dim, Number>::assemble_l2_0_product(
  const StateProductFactoryContext<dim, Number>& ctx,
  SparseMatrix<Number>& state_product_matrix) const
{
    auto l2_integrand =
    [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
        return fe.shape_value(i, q) * fe.shape_value(j, q);
    };

    Functions::ZeroFunction<dim> zero_bc_function(ctx.fe.n_components()); 
    uint32_t boundary_id = 0;
    AffineConstraints<Number> zero_BC_constraints;
    //DoFTools::make_hanging_node_constraints(ctx.dof_handler, zero_BC_constraints);

    for (const auto &id : ctx.dof_handler.get_triangulation().get_boundary_ids())
    {
        VectorTools::interpolate_boundary_values(
            ctx.dof_handler,
            id,
            zero_bc_function,
            zero_BC_constraints);
    }

    zero_BC_constraints.close();
    this->_assemble_product(ctx, state_product_matrix, l2_integrand, zero_BC_constraints);
}

template <int dim, typename Number>
void StateProductFactory<dim, Number>::assemble_h1_semi_product(
  const StateProductFactoryContext<dim, Number>& ctx,
  SparseMatrix<Number>& state_product_matrix) const
{
    auto h1_semi_integrand = 
    [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
      return fe.shape_grad(i, q) * fe.shape_grad(j, q);
    };
    AffineConstraints<Number> empty_BC_constraints;
    empty_BC_constraints.clear(); 
    empty_BC_constraints.close();

    this->_assemble_product(ctx, state_product_matrix, h1_semi_integrand, empty_BC_constraints);
}

template <int dim, typename Number>
void StateProductFactory<dim, Number>::assemble_h1_0_semi_product(
  const StateProductFactoryContext<dim, Number>& ctx,
  SparseMatrix<Number>& state_product_matrix) const
{
  auto h1_semi_integrand = 
    [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
      return fe.shape_grad(i, q) * fe.shape_grad(j, q);
    };

    Functions::ZeroFunction<dim> zero_bc_function(ctx.fe.n_components()); 
    uint32_t boundary_id = 0;
    AffineConstraints<Number> zero_BC_constraints;
    //DoFTools::make_hanging_node_constraints(ctx.dof_handler, zero_BC_constraints);

    for (const auto &id : ctx.dof_handler.get_triangulation().get_boundary_ids())
    {
        VectorTools::interpolate_boundary_values(
            ctx.dof_handler,
            id,
            zero_bc_function,
            zero_BC_constraints);
    }
    zero_BC_constraints.close();

    this->_assemble_product(ctx, state_product_matrix, h1_semi_integrand, zero_BC_constraints);
}

template <int dim, typename Number>
void StateProductFactory<dim, Number>::assemble_h1_product(
  const StateProductFactoryContext<dim, Number>& ctx,
  SparseMatrix<Number>& state_product_matrix) const
{
    auto h1_integrand = 
    [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
      return fe.shape_grad(i, q) * fe.shape_grad(j, q) + fe.shape_grad(i, q) * fe.shape_grad(j, q);;
    };

    AffineConstraints<Number> empty_BC_constraints;
    empty_BC_constraints.clear(); 
    empty_BC_constraints.close();

    this->_assemble_product(ctx, state_product_matrix, h1_integrand, empty_BC_constraints);
}

template <int dim, typename Number>
void StateProductFactory<dim, Number>::assemble_h1_0_product(
  const StateProductFactoryContext<dim, Number>& ctx,
  SparseMatrix<Number>& state_product_matrix) const
{
    auto h1_integrand = 
    [](unsigned int i, unsigned int j, unsigned int q, const FEValues<dim>& fe) {
      return fe.shape_grad(i, q) * fe.shape_grad(j, q) + fe.shape_grad(i, q) * fe.shape_grad(j, q);;
    };

    Functions::ZeroFunction<dim> zero_bc_function(ctx.fe.n_components()); 
    uint32_t boundary_id = 0;
    AffineConstraints<Number> zero_BC_constraints;
    //DoFTools::make_hanging_node_constraints(ctx.dof_handler, zero_BC_constraints);

    for (const auto &id : ctx.dof_handler.get_triangulation().get_boundary_ids())
    {
        VectorTools::interpolate_boundary_values(
            ctx.dof_handler,
            id,
            zero_bc_function,
            zero_BC_constraints);
    }
    zero_BC_constraints.close();

    this->_assemble_product(ctx, state_product_matrix, h1_integrand, zero_BC_constraints);
}

template <int dim, typename Number>
void StateProductFactory<dim, Number>::assemble_mass_product(
  const StateProductFactoryContext<dim, Number>& ctx,
  SparseMatrix<Number>& state_product_matrix) const
{
  StateProductFactory::assemble_l2_product(ctx, state_product_matrix);
}

template <int dim, typename Number>
void StateProductFactory<dim, Number>::assemble_boundary_mass_product(
  const StateProductFactoryContext<dim, Number>& ctx,
  SparseMatrix<Number>& state_product_matrix) const
{
  QGaussLobatto<dim-1> face_quadrature_formula(2);
  FEFaceValues<dim> face_fe_values(ctx.fe, face_quadrature_formula,
                                    update_values | update_gradients | update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = ctx.fe.dofs_per_cell;
  const unsigned int n_quadrature_points = face_quadrature_formula.size();

  AffineConstraints<Number> empty_BC_constraints;
  empty_BC_constraints.clear(); 
  empty_BC_constraints.close();

  FullMatrix<Number> boundary_cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  state_product_matrix.reinit(ctx.sparsity_pattern);
  state_product_matrix = 0;
  
  typename DoFHandler<dim>::active_cell_iterator cell = ctx.dof_handler.begin_active(), endc = ctx.dof_handler.end();
  for (; cell != endc; ++cell) {
    for (unsigned int face = 0; face < GeometryInfo<3>::faces_per_cell; face++) {
      if (cell->face(face)->at_boundary()) 
      {   
        boundary_cell_matrix = 0;
        cell->get_dof_indices(local_dof_indices);
        face_fe_values.reinit(cell, face);
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const unsigned int component_i = ctx.fe.system_to_component_index(i).first;

          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            const unsigned int component_j = ctx.fe.system_to_component_index(j).first;

            if (component_i != component_j)
            continue;

            for (unsigned int q_point = 0; q_point < n_quadrature_points; ++q_point) {
              boundary_cell_matrix(i, j) += 
              face_fe_values.shape_value(i,q_point)*
              face_fe_values.shape_value(j,q_point)*
              face_fe_values.JxW(q_point);
            }
          }
        }
        empty_BC_constraints.distribute_local_to_global(
          boundary_cell_matrix, 
          local_dof_indices, 
          state_product_matrix
        );
      }
    }
  }
  empty_BC_constraints.condense(state_product_matrix);
}



template <int dim, typename Number>
template <typename Integrand>
void StateProductFactory<dim, Number>::_assemble_product(
    const StateProductFactoryContext<dim, Number>& ctx,
    SparseMatrix<Number>& state_product_matrix,
    Integrand integrand,
    const AffineConstraints<Number>& constraints) const
{
  QGaussLobatto<3> quadrature_formula(2);
  FEValues<dim> fe_values(ctx.fe, quadrature_formula,
                          update_values | update_gradients | update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = ctx.fe.dofs_per_cell;
  const unsigned int n_quadrature_points = quadrature_formula.size();
  
  FullMatrix<Number> cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  state_product_matrix.reinit(ctx.sparsity_pattern);
  state_product_matrix = 0;

  typename DoFHandler<dim>::active_cell_iterator cell = ctx.dof_handler.begin_active(), endc = ctx.dof_handler.end();
  for (; cell != endc; ++cell) {
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell->get_dof_indices(local_dof_indices);

    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
      const unsigned int component_i = ctx.fe.system_to_component_index(i).first;

      for (unsigned int j = 0; j < dofs_per_cell; ++j) {
        const unsigned int component_j = ctx.fe.system_to_component_index(j).first;
        
        if (component_i != component_j)
          continue;

        for (unsigned int q_point = 0; q_point < n_quadrature_points; ++q_point) {
          cell_matrix(i, j) += integrand(i, j, q_point, fe_values) * fe_values.JxW(q_point);
          
        }
      }
    }
    constraints.distribute_local_to_global(cell_matrix, local_dof_indices, state_product_matrix);
  }

  constraints.condense(state_product_matrix);
}