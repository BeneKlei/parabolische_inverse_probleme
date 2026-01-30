#include <deal.II/base/exceptions.h>
#include <deal.II/lac/solver_cg.h>

#include <vector>

#include "MatrixOperator.hpp"

template class MatrixStack<double>;
template class MatrixOperator<double, SparseMatrix<double>>;
template class MatrixOperator<double, FullMatrix<double>>;

// ############################### MatrixStack ###############################

template <class Number>
MatrixStack<Number>::MatrixStack(
    std::vector<MatV>&& matrices, 
    bool affine)     
  : m_A(std::move(matrices))
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
void MatrixStack<Number>::materialize(
  MatV& matrix, 
  ArrayView<const float>& q,
  bool linear_part_only
) const
{
    AssertDimension(q.size(), dim_Q());

    matrix.reinit(m_A[0].get_sparsity_pattern());

    matrix = Number(0);
    if (m_affine)
    {        
        if (!linear_part_only) {
          matrix.add(1.0, m_A[0]);
        }
        
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
void MatrixStack<Number>::get_translation(
  MatV& matrix) const
{
  matrix.reinit(m_A[0].get_sparsity_pattern());
  if (m_affine)
  {
    matrix.add(1.0, m_A[0]);
  } 
  else 
  {
    matrix = Number(0);
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
  AssertDimension(v.size(), dim_V());

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

template <class Number, class MatrixType>
MatrixOperator<Number, MatrixType>::MatrixOperator(const MatrixOperator::MatV& matrix)
  : BaseOperator<Number>(true)
{
  // TODO Assuming mat is SPD. Add asserts for this.

  if constexpr (std::is_same_v<MatrixType, SparseMatrix<Number>>)
  {
    m_matrix.reinit(matrix.get_sparsity_pattern());
    m_matrix = Number(0);
    m_matrix.add(1.0, matrix);
  }
  else if constexpr (std::is_same_v<MatrixType, FullMatrix<Number>>)
  {
    m_matrix.reinit(matrix.m(), matrix.n());
    m_matrix = matrix;
  }
}

template <class Number, class MatrixType>
void MatrixOperator<Number, MatrixType>::apply(Vector<Number>       &y,
                                               const Vector<Number> &u) const
{
  AssertDimension(u.size(), dim_source());
  y.reinit(dim_range());
  m_matrix.vmult(y, u);
}

template <class Number, class MatrixType>
void MatrixOperator<Number, MatrixType>::apply_adjoint(Vector<Number>       &y,
                                                       const Vector<Number> &w) const
{
  AssertDimension(w.size(), dim_range());
  y.reinit(dim_source());
  m_matrix.Tvmult(y, w);
}

template <class Number, class MatrixType>
void MatrixOperator<Number, MatrixType>::apply_inverse(Vector<Number> &y,
                                                       const Vector<Number> &f) const
{
  //std::cout << "deal.ii apply_inverse" << std::endl;
  AssertDimension(f.size(), this->dim_range());
  y.reinit(this->dim_source());
  y = 0;

  SolverControl solver_control(20000, 1e-12);
  SolverCG<> solver(solver_control);

  if constexpr (std::is_same_v<MatrixType, SparseMatrix<Number>>)
  {
    PreconditionSSOR<SparseMatrix<Number>> preconditioner;
    preconditioner.initialize(m_matrix, 1.2);
    solver.solve(m_matrix, y, f, preconditioner);
  }
  else
  {
    PreconditionIdentity preconditioner;
    solver.solve(m_matrix, y, f, preconditioner);
  }
}

template <class Number, class MatrixType>
void MatrixOperator<Number, MatrixType>::apply_inverse_adjoint(Vector<Number>       &y,
                                                               const Vector<Number> &f) const
{
  this->apply_inverse(y,f);
}

template <class Number, class MatrixType>
std::unique_ptr<BaseOperator<Number>> 
MatrixOperator<Number, MatrixType>::jacobian(const Vector<Number> &u) const
{
  return std::make_unique<MatrixOperator<Number, MatrixType>>(
    m_matrix
  );
}

template <class Number, class MatrixType>
std::size_t MatrixOperator<Number, MatrixType>::dim_source() const 
{
  return m_matrix.n();
}

template <class Number, class MatrixType>
std::size_t MatrixOperator<Number, MatrixType>::dim_range() const 
{
  return m_matrix.m();
}

