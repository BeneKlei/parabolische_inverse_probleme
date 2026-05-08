#pragma once

#include <deal.II/lac/vector.h>
#include <deal.II/base/exceptions.h>

#include <memory>

using namespace dealii;

// Optional: for ops that exist only in special cases
DeclExceptionMsg(ExcNotDefined, "Operation not defined for this operator.");

// ============================================================================
// BaseOperator: pyMOR-like interface (apply / adjoint / inverse / inverse-adjoint)
// ============================================================================

template <class Number>
class BaseOperator
{
public:
  explicit BaseOperator(bool linear = false) : m_linear(linear) {}
  virtual ~BaseOperator() = default;

  // y = A u
  virtual void apply(Vector<Number>       &y,
                     const Vector<Number> &u) const = 0;

  // y = A^* u  (adjoint w.r.t. your chosen Hilbert identification)
  virtual void apply_adjoint(Vector<Number>       &y,
                             const Vector<Number> &u) const
  {
    AssertThrow(false, ExcNotDefined());
  }

  // y = A^{-1} u  (optional; may throw ExcNotDefined)
  virtual void apply_inverse(Vector<Number> &y,
                             const Vector<Number> &f,
                             double rtol,
                             double atol,
                             unsigned int maxiter) const
  {
    AssertThrow(false, ExcNotDefined());
  }

  // y = (A^*)^{-1} u  (pyMOR apply_inverse_adjoint semantics; optional)
  virtual void apply_inverse_adjoint(Vector<Number> &y,
                                     const Vector<Number> &f,
                                     double rtol,
                                     double atol,
                                     unsigned int maxiter) const
  {
    AssertThrow(false, ExcNotDefined());
  }

  virtual std::unique_ptr<BaseOperator<Number>> jacobian(const Vector<Number> &u) const
  {
    AssertThrow(false, ExcNotDefined());
    return nullptr;
  }

  virtual std::size_t dim_source() const = 0;
  virtual std::size_t dim_range() const = 0;

  const bool m_linear;
};
