#pragma once
#include <deal.II/lac/sparse_matrix.h>
#include <Eigen/Dense>
#include <memory>
#include <vector>

template <class Number>
class DenseBasis
{
public:
  DenseBasis(int n_full, int capacity);

  int n() const noexcept;
  int r() const noexcept;
  int capacity() const noexcept;

  void reserve(int new_cap);

  // Append dense block W (n×k), passed from numpy
  void append_from_numpy_ptr(const Number* W, int n, int k, int s0, int s1);

  // swap-remove active columns
  void remove_swap(const int* idxs, int m);

  Eigen::Ref<const Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>> active() const;
  Eigen::Ref<const Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>> active_left(int cols) const;
  Eigen::Ref<const Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>> active_block(int col0, int cols) const;

private:
  int n_ = 0;
  int r_ = 0;
  Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> V_; // n×cap
};

template <class Number>
class ReducedProjector
{
public:
  ReducedProjector(std::shared_ptr<DenseBasis<Number>> basis, int max_r, bool symmetric=true);

  int n() const noexcept;
  int r() const noexcept;
  int num_operators() const noexcept;

  // Store pointers to deal.II sparse matrices (no ownership)
  void set_operators(const std::vector<const dealii::SparseMatrix<Number>*>& ops);

  void notify_basis_changed();

  void project_full(int q);
  void project_full_all();
  void update_after_append(int k);

  void get(int q, Eigen::Ref<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>> out);

private:
  void spmm_(const dealii::SparseMatrix<Number>& A,
             const Eigen::Ref<const Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>>& X,
             Eigen::Ref<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>> Y) const;

  std::shared_ptr<DenseBasis<Number>> basis_;
  int max_r_;
  bool symmetric_;

  std::vector<const dealii::SparseMatrix<Number>*> ops_;
  std::vector<Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>> M_cache_;
  std::vector<unsigned char> valid_;
  int r_cached_ = 0;
  bool dirty_ = true;
};