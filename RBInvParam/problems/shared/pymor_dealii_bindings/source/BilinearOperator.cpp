#include <deal.II/base/exceptions.h>
#include <deal.II/lac/solver_cg.h>

#include <vector>

#include "BilinearOperator.hpp"

template class MatrixStack<double>;
template class BilinearAqOp<double>;

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
void MatrixStack<Number>::materialize(const Vector<Number> &q) const
{
    AssertThrow(q.size() == dim_Q(), ExcDimensionMismatch(q.size(), dim_Q()));

    if (m_has_cache && same_q(q))
        return;

    // Create/reinit cached matrix with same sparsity pattern as m_A[0]
    if (!m_cached_Aq)
    {
        m_cached_Aq = std::make_shared<MatV>();
        m_cached_Aq->reinit(m_sp);
    }

    if (m_affine)
    {        
        m_cached_Aq->add(1.0, m_A[0]);

        for (unsigned int i = 1; i < m_A.size(); ++i)
        if (q[i - 1] != Number(0))
            m_cached_Aq->add(q[i - 1], m_A[i]);
    }
    else
    {
        // A(q) = sum_i q[i] * Ai
        // BUG THis will not work
        *m_cached_Aq = Number(0);

        for (unsigned int i = 0; i < m_A.size(); ++i)
        if (q[i] != Number(0))
            m_cached_Aq->add(q[i], m_A[i]);
    }

    m_cached_q.reinit(q.size());
    m_cached_q = q;
    m_has_cache = true;
}

template <class Number>
void MatrixStack<Number>::clear_cache() const
{
    m_cached_Aq.reset();
    m_cached_q.reinit(0);
    m_has_cache = false;
}


template <class Number>
bool MatrixStack<Number>::same_q(const Vector<Number> &q) const
{
    if (!m_has_cache || m_cached_q.size() != q.size())
        return false;

    // TODO Use Deal.II methods if possible
    // exact compare; if you need tolerance, adjust here
    for (unsigned int i = 0; i < q.size(); ++i)
        if (m_cached_q[i] != q[i])
        return false;

    return true;
}

template <class Number>
bool MatrixStack<Number>::has_cache() const 
{ 
    return m_has_cache; 
}

template <class Number>
const Vector<Number>& MatrixStack<Number>::cached_q() const
{
  AssertThrow(m_has_cache, ExcNotDefined());
  return m_cached_q;
}

template <class Number>
std::shared_ptr<const typename MatrixStack<Number>::MatV> MatrixStack<Number>::cached_Aq() const
{
  AssertThrow(m_has_cache && m_cached_Aq, ExcNotDefined());
  return m_cached_Aq;
}


// ############################### BilinearAqOp ###############################

template <class Number>
BilinearAqOp<Number>::BilinearAqOp(std::shared_ptr<const Stack> stack,
                                   const Vector<Number>        &q)
  : m_stack(std::move(stack))
  , m_q(q)
{
  AssertThrow(m_stack != nullptr, ExcNotDefined());

  AssertDimension(m_q.size(), m_stack->dim_Q());

  m_stack->materialize(m_q);
  m_Aq = m_stack->cached_Aq();
  AssertThrow(m_Aq != nullptr, ExcNotDefined());

  // ---------------- Safeguards ----------------
  // The materialized matrix must come from the stack's cache for the same q.

  AssertThrow(m_stack->has_cache(), ExcNotDefined());

  // Ensure the pointer identity matches the stack cache.
  const auto cached_ptr = m_stack->cached_Aq();
  AssertThrow(cached_ptr != nullptr, ExcNotDefined());
  AssertThrow(cached_ptr.get() == m_Aq.get(), ExcNotDefined());

  // Ensure q matches cached_q exactly (use tolerance if needed).
  const Vector<Number> &cq = m_stack->cached_q();
  AssertDimension(cq.size(), m_q.size());
  for (unsigned int i = 0; i < m_q.size(); ++i)
    AssertThrow(cq[i] == m_q[i], ExcNotDefined());

  
}

template <class Number>
const Vector<Number>& BilinearAqOp<Number>::q() const
{
  return m_q;
}

template <class Number>
void BilinearAqOp<Number>::apply(Vector<Number>       &y,
                                 const Vector<Number> &u) const
{
  AssertDimension(u.size(), m_stack->dim_V());
  y.reinit(m_stack->dim_V());
  m_Aq->vmult(y, u);
}

template <class Number>
void BilinearAqOp<Number>::apply_adjoint(Vector<Number>       &y,
                                         const Vector<Number> &w) const
{
  AssertDimension(w.size(), m_stack->dim_V());
  y.reinit(m_stack->dim_V());
  m_Aq->Tvmult(y, w);
}

template <class Number>
void BilinearAqOp<Number>::apply_inverse(Vector<Number>       &y,
                                         const Vector<Number> &f) const
{
    std::cout << "Inner call" << std::endl;
    AssertDimension(f.size(), m_stack->dim_V());
    y.reinit(m_stack->dim_V());
    y = 0;

    SolverControl solver_control(20000, 1e-12);
    SolverCG<> solver(solver_control);
    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(*m_Aq, 1.2);
    solver.solve(*m_Aq, y, f, preconditioner);
}


template <class Number>
void BilinearAqOp<Number>::apply_inverse_adjoint(Vector<Number>       &y,
                                                 const Vector<Number> &f) const
{
  AssertThrow(false, ExcNotDefined());
}


template <class Number>
bool BilinearAqOp<Number>::has_inverse() const
{
  return true;
}


template <class Number>
bool BilinearAqOp<Number>::has_inverse_adjoint() const
{
  return false;
}

template <class Number>
std::size_t BilinearAqOp<Number>::dim_source() const 
{
  return m_stack->dim_V();
}

template <class Number>
std::size_t BilinearAqOp<Number>::dim_range() const 
{
  return m_stack->dim_V();
}

