#include <deal.II/numerics/fe_field_function.h>

#include "StoredEnergyFunction.hpp"

#include <cmath>

using namespace dealii;

// -------------------- ctor --------------------

template <int dim, typename Number>
NeoHookeanStoredEnergy<dim, Number>::NeoHookeanStoredEnergy(
    double mu, 
    double kappa,
    const FiniteElement<dim>& param_fe,
    const DoFHandler<dim>& param_dof_handler,
    const DoFHandler<dim>& state_dof_handler
)
  : m_mu(mu)
  , m_kappa(kappa)
  , m_beta((3.0 * kappa - 2.0 * mu) / (6.0 * mu))
  , m_c1(mu / 2.0)
  , m_param_fe(param_fe)
  , m_param_dof_handler(param_dof_handler)
  , m_state_dof_handler(state_dof_handler)
  , m_mapping()
{}


// template <int dim, typename Number>
// void NeoHookeanStoredEnergy<dim, Number>::setup_field_function()
// {
//   m_param_factor_field(
//       m_param_dof_handler, 
//       m_param,
//       m_mapping
//   );
// }


// -------------------- value --------------------

template <int dim, typename Number>
Number NeoHookeanStoredEnergy<dim, Number>::value(const PointType& p,
                                                  const TensorType& F) const
{
  const double param_factor = compute_param_factor(p);
  const double J  = determinant(F);
  const double I1 = trace(transpose(F) * F);

  const double value = m_c1 * (I1 - 3.0) 
                     + (m_c1 / m_beta) * (std::pow(J, -2.0 * m_beta) - 1.0);

  return param_factor * value;
}

// -------------------- gradient --------------------

template <int dim, typename Number>
typename NeoHookeanStoredEnergy<dim, Number>::TensorType
NeoHookeanStoredEnergy<dim, Number>::gradient(const PointType &p,
                                              const TensorType &F) const
{
  const double param_factor = compute_param_factor(p);
  const double J = determinant(F);

  TensorType grad;
  const double eps = 1e-12;
  if (std::abs(J) < eps)
    return grad;

  TensorType JY = invert(transpose(F));     // F^{-T}
  JY *= std::pow(J, -2.0 * m_beta);

  grad  = F;
  grad -= JY;
  grad *= (2.0 * m_c1);

  return param_factor * grad;
}

// -------------------- hessian --------------------
//
// NOTE: True Hessian is 4th-order. This exists only to satisfy the interface.

template <int dim, typename Number>
typename NeoHookeanStoredEnergy<dim, Number>::TensorType
NeoHookeanStoredEnergy<dim, Number>::hessian(const PointType & /*p*/,
                                             const TensorType & /*F*/) const
{
  return TensorType(); // zero
}

// -------------------- contracted_hessian (full H) --------------------

template <int dim, typename Number>
typename NeoHookeanStoredEnergy<dim, Number>::TensorType
NeoHookeanStoredEnergy<dim, Number>::contracted_hessian(const PointType &p,
                                                        const TensorType &F,
                                                        const TensorType &H) const
{
  const double param_factor = compute_param_factor(p);
  const double J = determinant(F);

  TensorType cont;
  const double eps = 1e-12;
  if (std::abs(J) < eps)
  {
    cont = H;
    cont *= (2.0 * m_c1);
    return param_factor * cont;
  }

  const TensorType F_inv_T = invert(transpose(F));
  const double     Jfac    = std::pow(J, -2.0 * m_beta);

  const TensorType FHF = F_inv_T * transpose(H) * F_inv_T;

  double inner = 0.0;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      inner += F_inv_T[i][j] * H[i][j]; // <F^{-T}, H>

  cont = H;
  cont += (Jfac * FHF);

  TensorType extra = F_inv_T;
  extra *= (Jfac * (2.0 * m_beta) * inner);
  cont += extra;

  cont *= (2.0 * m_c1);
  return param_factor * cont;
}

// -------------------- contracted_hessian (row-comp overload) --------------------

template <int dim, typename Number>
typename NeoHookeanStoredEnergy<dim, Number>::TensorType
NeoHookeanStoredEnergy<dim, Number>::contracted_hessian(const PointType &p,
                                                        const TensorType &F,
                                                        const TensorTypeMinus1 &H,
                                                        std::size_t comp) const
{
  AssertIndexRange(comp, dim);

  TensorType test_H;
  for (unsigned int j = 0; j < dim; ++j)
    test_H[comp][j] = H[j];

  return this->contracted_hessian(p, F, test_H);
}

// -------------------- utils --------------------

template <int dim, typename Number>
double NeoHookeanStoredEnergy<dim, Number>::compute_param_factor(const PointType& p) const
{
  // if (!in_first_layer(p))
  //   return 1.0;
  
  // return m_param_factor_field.value(p);
  return 1.0;
}

template <int dim, typename Number>
bool NeoHookeanStoredEnergy<dim, Number>::in_first_layer(const PointType& p) const
{
    const auto cell_and_ref = GridTools::find_active_cell_around_point(
      m_mapping, 
      m_param_dof_handler, 
      p
    );
    const auto &cell = cell_and_ref.first;

    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 1)
            return true;

    return false;
}

// -------------------- explicit instantiations --------------------

template class NeoHookeanStoredEnergy<2, double>;
template class NeoHookeanStoredEnergy<3, double>;

