#include <deal.II/numerics/fe_field_function.h>

#include "StoredEnergyFunction.hpp"

#include <cmath>

using namespace dealii;

// ----------------------------------- HookeanStoredEnergy -----------------------------------

template <int dim>
HookeanStoredEnergy<dim>::HookeanStoredEnergy(
    double mu, 
    double lambda
)
  : m_lambda(lambda)
  , m_mu(mu)
{}

// -------------------- value --------------------

template <int dim>
double HookeanStoredEnergy<dim>::value(const PointType& p,
                                       const TensorType& F) const
{
  const Tensor<2,dim> I = Tensor<2, dim>(unit_symmetric_tensor<dim, double>());
  const Tensor<2,dim> Y = F - I;
  const double trY = trace(Y);

  const SymmetricTensor<2, dim, double> Y_sym = symmetrize(Y);
  const double value = 0.5 * m_lambda * trY * trY + m_mu * Tensor<2, dim>(Y_sym).norm_square();

  return value;
}

// -------------------- gradient --------------------

template <int dim>
typename HookeanStoredEnergy<dim>::TensorType
HookeanStoredEnergy<dim>::gradient(const PointType &p,
                                   const TensorType &F) const
{
  TensorType grad;
  grad = 0.0;

  const Tensor<2,dim> I = Tensor<2, dim>(unit_symmetric_tensor<dim, double>());
  const Tensor<2,dim> Y = F - I;
  const double trY = trace(Y);

  const SymmetricTensor<2, dim, double> Y_sym = symmetrize(Y);

  grad = m_lambda * trY * Tensor<2, dim>(unit_symmetric_tensor<dim, double>());
  grad += 2.0 * m_mu * Tensor<2, dim>(Y_sym);
  return grad;
}

// -------------------- hessian --------------------
//
// NOTE: True Hessian is 4th-order. This exists only to satisfy the interface.

template <int dim>
typename HookeanStoredEnergy<dim>::TensorType
HookeanStoredEnergy<dim>::hessian(const PointType & /*p*/,
                                  const TensorType & /*F*/) const
{
  return TensorType(); // zero
}

// -------------------- contracted_hessian (full H) --------------------

template <int dim>
typename HookeanStoredEnergy<dim>::TensorType
HookeanStoredEnergy<dim>::contracted_hessian(const PointType &p,
                                             const TensorType &F,
                                             const TensorType &H) const
{
  TensorType cont;
  cont = 0.0;

  const double trH = trace(H);
  const SymmetricTensor<2, dim> H_sym = symmetrize(H);

  cont = m_lambda * trH * Tensor<2, dim>(unit_symmetric_tensor<dim, double>());
  cont += 2.0 * m_mu * Tensor<2, dim, double>(H_sym);

  return cont;
}

// -------------------- contracted_hessian (row-comp overload) --------------------

template <int dim>
typename HookeanStoredEnergy<dim>::TensorType
HookeanStoredEnergy<dim>::contracted_hessian(const PointType &p,
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

// -------------------- explicit instantiations --------------------

template class HookeanStoredEnergy<2>;
template class HookeanStoredEnergy<3>;

// ---------------------------------- NeoHookeanStoredEnergy ----------------------------------
// -------------------- ctor --------------------

template <int dim>
NeoHookeanStoredEnergy<dim>::NeoHookeanStoredEnergy(
    double mu, 
    double kappa
)
  : m_mu(mu)
  , m_kappa(kappa)
  , m_beta((3.0 * kappa - 2.0 * mu) / (6.0 * mu))
  , m_c1(mu / 2.0)
{}

// -------------------- value --------------------

template <int dim>
double NeoHookeanStoredEnergy<dim>::value(const PointType& p,
                                          const TensorType& F) const
{
  const double D  = determinant(F);
  const double I1 = trace(transpose(F) * F);

  const double value = m_c1 * (I1 - 3.0) 
                     + (m_c1 / m_beta) * (std::pow(D, -2.0 * m_beta) - 1.0);

  return value;
}

// -------------------- gradient --------------------

template <int dim>
typename NeoHookeanStoredEnergy<dim>::TensorType
NeoHookeanStoredEnergy<dim>::gradient(const PointType &p,
                                      const TensorType &F) const
{
  const double D  = determinant(F);

  // const Tensor<2,dim> I = Tensor<2, dim>(unit_symmetric_tensor<dim, double>());
  // const Tensor<2,dim> Y = F - I;

  TensorType grad;
  grad = 0.0;

  if (std::abs(D) < m_tol_gradient)
    return grad;

  grad = F;
  grad -= (std::pow(D, -2.0 * m_beta) * invert(transpose(F)));
  grad *= (2.0 * m_c1);

  // std::cout << Y.norm() << std::endl;
  // std::cout << invert(transpose(Y)) << std::endl;
  // std::cout << grad << std::endl;

  return grad;
}

// -------------------- hessian --------------------
//
// NOTE: True Hessian is 4th-order. This exists only to satisfy the interface.

template <int dim>
typename NeoHookeanStoredEnergy<dim>::TensorType
NeoHookeanStoredEnergy<dim>::hessian(const PointType & /*p*/,
                                     const TensorType & /*F*/) const
{
  return TensorType(); // zero
}

// -------------------- contracted_hessian (full H) --------------------

template <int dim>
typename NeoHookeanStoredEnergy<dim>::TensorType
NeoHookeanStoredEnergy<dim>::contracted_hessian(const PointType &p,
                                                const TensorType &F,
                                                const TensorType &H) const
{
  const double D  = determinant(F);
  // const Tensor<2,dim> I = Tensor<2, dim>(unit_symmetric_tensor<dim, double>());
  // const Tensor<2,dim> Y = F - I;

  TensorType cont;
  cont = 0.0;
  
  if (D < m_tol_hessian)
  {
    cont = H;
    cont *= (2.0 * m_c1);
    return cont;
  }

  const TensorType F_inv_T = invert(transpose(F));
  const double     Dfac    = std::pow(D, -2.0 * m_beta);

  cont = H;
  cont += (Dfac * F_inv_T * transpose(H) * F_inv_T);
  cont += (2.0 * m_beta * Dfac * scalar_product(F_inv_T, H) * F_inv_T);
  cont *= (2.0 * m_c1);

  return cont;
}

// -------------------- contracted_hessian (row-comp overload) --------------------

template <int dim>
typename NeoHookeanStoredEnergy<dim>::TensorType
NeoHookeanStoredEnergy<dim>::contracted_hessian(const PointType &p,
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

// -------------------- explicit instantiations --------------------

template class NeoHookeanStoredEnergy<2>;
template class NeoHookeanStoredEnergy<3>;

