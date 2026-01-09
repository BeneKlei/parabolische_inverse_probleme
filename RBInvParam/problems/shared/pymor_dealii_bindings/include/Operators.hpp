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
  virtual ~BaseOperator() = default;

  // y = A x
  virtual void apply(Vector<Number>       &y,
                     const Vector<Number> &x) const = 0;

  // y = A^* x  (adjoint w.r.t. your chosen Hilbert identification)
  virtual void apply_adjoint(Vector<Number>       &y,
                             const Vector<Number> &x) const = 0;

  // y = A^{-1} x  (optional; may throw ExcNotDefined)
  virtual void apply_inverse(Vector<Number>       &y,
                             const Vector<Number> &x) const
  {
    AssertThrow(false, ExcNotDefined());
  }

  // y = (A^*)^{-1} x  (pyMOR apply_inverse_adjoint semantics; optional)
  virtual void apply_inverse_adjoint(Vector<Number>       &y,
                                     const Vector<Number> &x) const
  {
    AssertThrow(false, ExcNotDefined());
  }

  virtual bool has_inverse() const { return false; }
  virtual bool has_inverse_adjoint() const { return false; }

  virtual std::size_t dim_source() const = 0;
  virtual std::size_t dim_range() const = 0;

};

// // Convenience aliases (storage is the same Vector<Number>; meaning is by convention)
// template <class Number>
// using OpVtoVdual = BaseOperator<Number>; // V -> V' (V' represented in V)

// template <class Number>
// using OpQtoVdual = BaseOperator<Number>; // Q -> V' (V' represented in V)


// ============================================================================
// Split operators (interfaces only)
// - A_op(q,u):     operator in u, i.e. u ↦ A(q,u)     (V -> V')
// - dA_dq_op(q,u): operator in dq, i.e. dq ↦ ∂_qA(q,u)[dq] (Q -> V')
// - dA_du_op(q,u): operator in du, i.e. du ↦ ∂_uA(q,u)[du] (V -> V')
// ============================================================================

// template <class Number>
// class Aq_op : public OpVtoVdual<Number>
// {
// public:
//   virtual ~Aq_op() = default;
//   virtual const Vector<Number> &q() const = 0;
// };

// template <class Number>
// class dAqu_dq_op : public OpQtoVdual<Number>
// {
// public:
//   virtual ~dAqu_dq_op() = default;

//   virtual const Vector<Number> &q() const = 0;
//   virtual const Vector<Number> &u() const = 0;
// };

// template <class Number>
// class dAqu_du_op : public OpVtoVdual<Number>
// {
// public:
//   virtual ~dAqu_du_op() = default;

//   virtual const Vector<Number> &q() const = 0;
//   virtual const Vector<Number> &u() const = 0;
// };









// // // ============================================================================
// // // Family interface (creates the three operators; all may share one backend/tensor)
// // // ============================================================================

// template <class Number>
// class ParametricOperatorFamily
// {
// public:
//   virtual ~ParametricOperatorFamily() = default;

//   // u ↦ A(q,u)  (linear operator in "u" at (q,u); for bilinear, depends only on q)
//   virtual std::shared_ptr<const A_op<Number>>
//   op(const Vector<Number> &q, const Vector<Number> &u) const = 0;

//   // dq ↦ ∂_q A(q,u)[dq]
//   virtual std::shared_ptr<const dA_dq_op<Number>>
//   d_q(const Vector<Number> &q, const Vector<Number> &u) const = 0;

//   // du ↦ ∂_u A(q,u)[du]
//   virtual std::shared_ptr<const dA_du_op<Number>>
//   d_u(const Vector<Number> &q, const Vector<Number> &u) const = 0;

//   // Dimensions for checks
//   virtual unsigned int dim_Q() const = 0;
//   virtual unsigned int dim_V() const = 0;
// };


// ============================================================================