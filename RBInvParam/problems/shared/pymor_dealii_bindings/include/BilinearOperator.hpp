#pragma once

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <memory>
#include <vector>

#include "Operators.hpp"

using namespace dealii;

// ############################### MatrixStack ###############################

template <class Number>
class MatrixStack
{
public:
  using MatV = SparseMatrix<Number>;

  MatrixStack(
    std::vector<MatV>&& matrices, 
    const SparsityPattern& sp,
    bool affine = false
  );

  unsigned int dim_Q() const;
  unsigned int dim_V() const;

  const MatV &A_i(unsigned int i) const;

  // Returns shared matrix for this q (cached). May assemble if not cached.
  void materialize(const Vector<Number> &q) const;

  void clear_cache() const;
  bool same_q(const Vector<Number> &q) const;
  bool has_cache() const;
  const Vector<Number>& cached_q() const;
  std::shared_ptr<const MatV> cached_Aq() const;

private:
  std::vector<MatV> m_A;
  const SparsityPattern& m_sp;

  bool m_affine;

  // cache (one-entry or multi-entry). Marked mutable since materialize() is logically const.
  mutable std::shared_ptr<MatV> m_cached_Aq;
  mutable Vector<Number>        m_cached_q;
  mutable bool                  m_has_cache = false;
};

// ############################### BilinearAqOp ###############################

template <class Number>
class BilinearAqOp : public Aq_op<Number>
{
public:
  using Stack = MatrixStack<Number>;
  using MatV  = typename Stack::MatV;

  BilinearAqOp(std::shared_ptr<const Stack> stack,
               const Vector<Number>&        q);

  const Vector<Number> &q() const override;

  void apply(Vector<Number> &y,
             const Vector<Number> &u) const override;

  void apply_adjoint(Vector<Number> &y,
                     const Vector<Number> &w) const override;

  void apply_inverse(Vector<Number> &y,
                     const Vector<Number> &f) const override;

  void apply_inverse_adjoint(Vector<Number> &y,
                             const Vector<Number> &f) const override;

  bool has_inverse() const override;
  bool has_inverse_adjoint() const override;

  std::size_t dim_source() const override;
  std::size_t dim_range() const override;

private:
  std::shared_ptr<const Stack> m_stack;
  Vector<Number>               m_q;
  std::shared_ptr<const MatV>  m_Aq;
};