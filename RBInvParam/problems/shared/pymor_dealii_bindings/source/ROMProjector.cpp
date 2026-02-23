// ROMProjector.cpp
#include "ROMProjector.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <stdexcept>

#include <deal.II/lac/vector.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

// IMPORTANT:
// These are templates defined in a .cpp, so we MUST explicitly instantiate
// for the Number types you use (likely double). See bottom of file.

// ========================= DenseBasis<Number> =========================

template <class Number>
DenseBasis<Number>::DenseBasis(int n_full, int capacity)
  : n_(n_full), r_(0)
{
  if (n_ <= 0)
    throw std::runtime_error("DenseBasis: n_full must be positive");
  if (capacity < 0)
    throw std::runtime_error("DenseBasis: capacity must be non-negative");

  V_.resize(n_, capacity);
  V_.setZero();
}

template <class Number>
int DenseBasis<Number>::n() const noexcept { return n_; }

template <class Number>
int DenseBasis<Number>::r() const noexcept { return r_; }

template <class Number>
int DenseBasis<Number>::capacity() const noexcept { return static_cast<int>(V_.cols()); }

template <class Number>
void DenseBasis<Number>::reserve(int new_cap)
{
  if (new_cap <= capacity())
    return;

  Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> V_new(n_, new_cap);
  V_new.setZero();
  if (r_ > 0)
    V_new.leftCols(r_) = V_.leftCols(r_);

  V_.swap(V_new);
}

template <class Number>
void DenseBasis<Number>::append_from_numpy_ptr(const Number* W, int n, int k, int s0, int s1)
{
  if (!W)
    throw std::runtime_error("DenseBasis.append_from_numpy_ptr: W is null");
  if (n != n_)
    throw std::runtime_error("DenseBasis.append_from_numpy_ptr: wrong row dimension");
  if (k <= 0)
    return;
  if (s0 <= 0 || s1 <= 0)
    throw std::runtime_error("DenseBasis.append_from_numpy_ptr: invalid strides");

  if (r_ + k > capacity()) {
    const int grown = std::max(r_ + k, std::max(2 * capacity(), 16));
    reserve(grown);
  }

  // W(i,j) is at W + i*s0 + j*s1 (strides are in units of Number)
  for (int j = 0; j < k; ++j) {
    const Number* col = W + j * s1;
    for (int i = 0; i < n_; ++i) {
      V_(i, r_ + j) = col[i * s0];
    }
  }
  r_ += k;
}

template <class Number>
void DenseBasis<Number>::remove_swap(const int* idxs, int m)
{
  if (!idxs)
    throw std::runtime_error("DenseBasis.remove_swap: idxs is null");
  if (m <= 0)
    return;

  for (int t = 0; t < m; ++t) {
    const int i = idxs[t];
    if (i < 0 || i >= r_)
      throw std::runtime_error("DenseBasis.remove_swap: index out of range");

    const int last = r_ - 1;
    if (i != last)
      V_.col(i).swap(V_.col(last));

    // Optional: clear freed column
    V_.col(last).setZero();
    r_ -= 1;
  }
}

template <class Number>
Eigen::Ref<const Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>>
DenseBasis<Number>::active() const
{
  return V_.leftCols(r_);
}

template <class Number>
Eigen::Ref<const Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>>
DenseBasis<Number>::active_left(int cols) const
{
  if (cols < 0 || cols > r_)
    throw std::runtime_error("DenseBasis.active_left: cols out of range");
  return V_.leftCols(cols);
}

template <class Number>
Eigen::Ref<const Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>>
DenseBasis<Number>::active_block(int col0, int cols) const
{
  if (col0 < 0 || cols < 0 || col0 + cols > r_)
    throw std::runtime_error("DenseBasis.active_block: range out of bounds");
  return V_.block(0, col0, n_, cols);
}

// ====================== ReducedProjector<Number> ======================

template <class Number>
ReducedProjector<Number>::ReducedProjector(std::shared_ptr<DenseBasis<Number>> basis,
                                           int max_r,
                                           bool symmetric)
  : basis_(std::move(basis))
  , max_r_(max_r)
  , symmetric_(symmetric)
  , ops_()
  , M_cache_()
  , valid_()
  , r_cached_(0)
  , dirty_(true)
{
  if (!basis_)
    throw std::runtime_error("ReducedProjector: basis is null");
  if (max_r_ <= 0)
    throw std::runtime_error("ReducedProjector: max_r must be positive");
}

template <class Number>
int ReducedProjector<Number>::n() const noexcept { return basis_->n(); }

template <class Number>
int ReducedProjector<Number>::r() const noexcept { return basis_->r(); }

template <class Number>
int ReducedProjector<Number>::num_operators() const noexcept { return static_cast<int>(ops_.size()); }

template <class Number>
void ReducedProjector<Number>::set_operators(const std::vector<const dealii::SparseMatrix<Number>*>& ops)
{
  ops_ = ops;
  const std::size_t Q = ops_.size();

  M_cache_.clear();
  valid_.assign(Q, 0);

  M_cache_.reserve(Q);
  for (std::size_t q = 0; q < Q; ++q) {
    if (!ops_[q])
      throw std::runtime_error("ReducedProjector.set_operators: null matrix pointer");

    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> M(max_r_, max_r_);
    M.setZero();
    M_cache_.push_back(std::move(M));
  }

  dirty_ = true;
  r_cached_ = 0;
}

template <class Number>
void ReducedProjector<Number>::notify_basis_changed()
{
  dirty_ = true;
  std::fill(valid_.begin(), valid_.end(), 0);
  r_cached_ = 0;
}

template <class Number>
void ReducedProjector<Number>::spmm_(
  const dealii::SparseMatrix<Number>& A,
  const Eigen::Ref<const Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>>& X,
  Eigen::Ref<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>> Y) const
{
  const int n_full = n();

  if ((int)A.m() != n_full || (int)A.n() != n_full)
    throw std::runtime_error("spmm_: matrix dimension mismatch");
  if (X.rows() != n_full || Y.rows() != n_full)
    throw std::runtime_error("spmm_: row mismatch");
  if (Y.cols() != X.cols())
    throw std::runtime_error("spmm_: col mismatch");

  const int k = (int)X.cols();

  // NOTE on threading:
  // - If you parallelize over operators q in project_full_all/update_after_append,
  //   then REMOVE OpenMP from here to avoid oversubscription.
  // - As written, this is parallel over RHS columns.
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    dealii::Vector<Number> in(n_full), out(n_full);

#ifdef _OPENMP
#pragma omp for
#endif
    for (int j = 0; j < k; ++j) {
      for (int i = 0; i < n_full; ++i)
        in[i] = X(i, j);

      A.vmult(out, in);

      for (int i = 0; i < n_full; ++i)
        Y(i, j) = out[i];
    }
  }
}

template <class Number>
void ReducedProjector<Number>::project_full(int q)
{
  if (q < 0 || q >= (int)ops_.size())
    throw std::runtime_error("project_full: operator index out of range");

  const int r_now = r();
  if (r_now > max_r_)
    throw std::runtime_error("project_full: basis r exceeds max_r");

  const auto& A = *ops_[q];
  const auto V = basis_->active(); // n×r

  Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> T(n(), r_now);
  spmm_(A, V, T);

  Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> M = V.transpose() * T;

  M_cache_[q].topLeftCorner(r_now, r_now) = M;
  valid_[q] = 1;

  dirty_ = false;
  r_cached_ = r_now;
}

template <class Number>
void ReducedProjector<Number>::project_full_all()
{
  const int r_now = r();
  if (r_now > max_r_)
    throw std::runtime_error("project_full_all: basis r exceeds max_r");
  if (ops_.empty())
    return;

  // If spmm_ is threaded, prefer running this loop sequentially.
  for (int q = 0; q < (int)ops_.size(); ++q) {
    const auto& A = *ops_[q];
    const auto V = basis_->active();

    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> T(n(), r_now);
    spmm_(A, V, T);

    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> M = V.transpose() * T;

    M_cache_[q].topLeftCorner(r_now, r_now) = M;
    valid_[q] = 1;
  }

  dirty_ = false;
  r_cached_ = r_now;
}

template <class Number>
void ReducedProjector<Number>::update_after_append(int k)
{
  if (k <= 0)
    return;

  const int r_now = r();
  if (r_now > max_r_)
    throw std::runtime_error("update_after_append: basis r exceeds max_r");

  const int r0 = r_now - k;
  if (r0 < 0)
    throw std::runtime_error("update_after_append: k too large");

  if (dirty_ || r_cached_ != r0) {
    project_full_all();
    return;
  }
  if (ops_.empty())
    return;

  const auto V = basis_->active_left(r0);     // n×r0
  const auto W = basis_->active_block(r0, k); // n×k

  for (int q = 0; q < (int)ops_.size(); ++q) {
    if (!valid_[q])
      continue;

    const auto& A = *ops_[q];

    // TW = A * W
    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> TW(n(), k);
    spmm_(A, W, TW);

    // Blocks
    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> VTAW = V.transpose() * TW; // r0×k
    Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> WTAW = W.transpose() * TW; // k×k

    auto& M = M_cache_[q];
    M.block(0,  r0, r0, k) = VTAW;

    if (symmetric_) {
      M.block(r0, 0, k, r0) = VTAW.transpose();
    } else {
      // Non-symmetric: compute WTAV explicitly
      Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> TV(n(), r0);
      spmm_(A, V, TV);
      Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> WTAV = W.transpose() * TV; // k×r0
      M.block(r0, 0, k, r0) = WTAV;
    }

    M.block(r0, r0, k, k) = WTAW;
  }

  dirty_ = false;
  r_cached_ = r_now;
}

template <class Number>
void ReducedProjector<Number>::get(
  int q,
  Eigen::Ref<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>> out)
{
  if (q < 0 || q >= (int)ops_.size())
    throw std::runtime_error("get: operator index out of range");

  const int r_now = r();
  if (r_now > max_r_)
    throw std::runtime_error("get: basis r exceeds max_r");
  if (out.rows() != r_now || out.cols() != r_now)
    throw std::runtime_error("get: out matrix has wrong shape");

  if (dirty_ || r_cached_ != r_now || !valid_[q]) {
    project_full(q);
  }

  out = M_cache_[q].topLeftCorner(r_now, r_now);
}

// ======================= Explicit instantiation =======================
// Because the template definitions are in this .cpp, we instantiate the types we need.
template class DenseBasis<double>;
template class ReducedProjector<double>;