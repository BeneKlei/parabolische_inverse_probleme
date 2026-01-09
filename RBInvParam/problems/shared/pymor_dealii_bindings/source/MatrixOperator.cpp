#include <deal.II/base/exceptions.h>
#include <deal.II/lac/solver_cg.h>

#include <vector>

#include "MatrixOperator.hpp"

template class MatrixStack<double>;
template class MatrixOperator<double>;

// ############################### MatrixStack ###############################

template <class Number>
MatrixStack<Number>::MatrixStack(
    std::vector<MatV>&& matrices, 
    const SparsityPattern& sp,
    bool affine) 
  : m_A(std::move(matrices))
  , m_sp(sp)
  , m_affine(affine)
{
  AssertThrow(!m_A.empty(), ExcMessage("MatrixStack: empty m_A."));
}

template <class Number>
unsigned int MatrixStack<Number>::dim_Q() const
{
  AssertThrow(m_A.size() > 0, ExcInternalError());
  return m_affine ? m_A.size() - 1 : m_A.size();
}

template <class Number>
unsigned int MatrixStack<Number>::dim_V() const
{
    return m_A[0].m();
}

template <class Number>
const typename MatrixStack<Number>::MatV &
MatrixStack<Number>::A_i(const unsigned int i) const
{
  AssertIndexRange(i, m_A.size());
  return m_A[i];
}

template <class Number>
void MatrixStack<Number>::materialize(MatV& matrix, ArrayView<const float>& q) const
{
    AssertThrow(q.size() == dim_Q(), ExcDimensionMismatch(q.size(), dim_Q()));

    matrix.reinit(m_sp);

    matrix = Number(0);
    if (m_affine)
    {        
        matrix.add(1.0, m_A[0]);

        for (unsigned int i = 1; i < m_A.size(); ++i)
        if (q[i - 1] != Number(0))
            matrix.add(q[i - 1], m_A[i]);
    }
    else
    {
        for (unsigned int i = 0; i < m_A.size(); ++i)
        if (q[i] != Number(0))
            matrix.add(q[i], m_A[i]);
    }

}

template <class Number>
void
MatrixStack<Number>::apply_to_each_matrix(
  const Vector<Number> &v,
  std::vector<Vector<Number>> &result,
  const bool include_affine_base) const
{
  AssertThrow(dim_V() > 0, ExcInternalError());
  AssertThrow(v.size() == dim_V(), ExcDimensionMismatch(v.size(), dim_V()));

  const unsigned int start =
    (m_affine && !include_affine_base) ? 1u : 0u;

  const unsigned int n_matrices = m_A.size() - start;

  // Resize output buffer
  result.resize(n_matrices);

  for (unsigned int k = 0; k < n_matrices; ++k)
  {
    const unsigned int i = start + k;

    // Ensure correct size
    result[k].reinit(dim_V());

    // result[k] = A_i * v
    m_A[i].vmult(result[k], v);
  }
}

// ############################### MatrixOperator ###############################

template <class Number>
MatrixOperator<Number>::MatrixOperator(MatrixOperator::MatV matrix, const SparsityPattern& sp)
  : m_matrix(std::move(matrix))
  , m_sp(sp) {}


template <class Number>
void MatrixOperator<Number>::apply(Vector<Number>       &y,
                                   const Vector<Number> &u) const
{
  AssertDimension(u.size(), this->dim_source());
  y.reinit(this->dim_range());
  m_matrix.vmult(y, u);
}

template <class Number>
void MatrixOperator<Number>::apply_adjoint(Vector<Number>       &y,
                                           const Vector<Number> &w) const
{
  AssertDimension(w.size(), this->dim_range());
  y.reinit(this->dim_source());
  m_matrix.Tvmult(y, w);
}

template <class Number>
void MatrixOperator<Number>::apply_inverse(Vector<Number>       &y,
                                           const Vector<Number> &f) const
{
    AssertDimension(f.size(), this->dim_range());
    y.reinit(this->dim_source());
    y = 0;

    SolverControl solver_control(20000, 1e-12);
    SolverCG<> solver(solver_control);
    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(m_matrix, 1.2);
    solver.solve(m_matrix, y, f, preconditioner);
}


template <class Number>
void MatrixOperator<Number>::apply_inverse_adjoint(Vector<Number>       &y,
                                                   const Vector<Number> &f) const
{
  AssertThrow(false, ExcNotDefined());
}

template <class Number>
bool MatrixOperator<Number>::has_inverse() const
{
  return true;
}

template <class Number>
bool MatrixOperator<Number>::has_inverse_adjoint() const
{
  return false;
}

template <class Number>
std::size_t MatrixOperator<Number>::dim_source() const 
{
  return m_matrix.m();
}

template <class Number>
std::size_t MatrixOperator<Number>::dim_range() const 
{
  return m_matrix.n();
}

