#pragma once

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>

#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/base/array_view.h>

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
    bool affine = false
  );

  unsigned int dim_Q() const;
  unsigned int dim_V() const;

  const MatV &A_i(unsigned int i) const;


  void materialize(MatV& matrix,
                   ArrayView<const float>& q,
                   bool linear_part_only = false) const;

  void apply_to_each_matrix(const Vector<Number> &v,
                            std::vector<Vector<Number>> &result,
                            const bool include_affine_base) const;

  void get_translation(MatV& matrix) const;

private:
  std::vector<MatV> m_A;
  bool m_affine;
  SparsityPattern m_sp;

  // cache (one-entry or multi-entry). Marked mutable since materialize() is logically const.
  // mutable std::shared_ptr<MatV> m_cached_Aq;
  // mutable Vector<Number>        m_cached_q;
  // mutable bool                  m_has_cache = false;
};

// ############################### MatrixOperator ###############################
 
template <class Number,  class MatrixType>
class MatrixOperator : public BaseOperator<Number>
{
public:
  using MatV = MatrixType;

  MatrixOperator(const MatV& matrix);

  void apply(Vector<Number> &y,
             const Vector<Number> &u) const override;

  void apply_adjoint(Vector<Number> &y,
                     const Vector<Number> &w) const override;

  void apply_inverse(Vector<Number> &y,
                     const Vector<Number> &f) const override;

  void apply_inverse_adjoint(Vector<Number> &y,
                             const Vector<Number> &f) const override;
  
  std::unique_ptr<BaseOperator<Number>> jacobian(const Vector<Number> &u) const override;
  
  const MatV& get_matrix() const { return m_matrix; }

  std::size_t dim_source() const override;
  std::size_t dim_range() const override;

private:
  MatV  m_matrix;
};


template <class Number>
using SparseMatrixOperator = MatrixOperator<Number, SparseMatrix<Number>>;

template <class Number>
using FullMatrixOperator   = MatrixOperator<Number, FullMatrix<Number>>;