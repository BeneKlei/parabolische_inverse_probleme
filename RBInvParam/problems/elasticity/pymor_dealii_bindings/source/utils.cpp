#include "utils.hpp"

std::shared_ptr<SparsityPattern> make_product_sparsity_AB(const SparseMatrix<double>& A,
                                         const SparseMatrix<double>& B) {
    AssertThrow(A.n() == B.m(), ExcDimensionMismatch(A.n(), B.m()));
    DynamicSparsityPattern dsp(A.m(), B.n());
    for (unsigned int i = 0; i < A.m(); ++i) {
        for (auto a = A.begin(i); a != A.end(i); ++a) {
            const unsigned int k = a->column();
            for (auto b = B.begin(k); b != B.end(k); ++b) {
                dsp.add(i, b->column());
            }
        }
    }
    auto sp = std::make_shared<SparsityPattern>();
    sp->copy_from(dsp);
    return sp;
}

std::shared_ptr<SparsityPattern> make_product_sparsity_ATB(const SparseMatrix<double>& A,
                                          const SparseMatrix<double>& B) {
  // Build sparsity of A^T * B
  AssertThrow(A.m() == B.m(), ExcDimensionMismatch(A.m(), B.m()));
  const unsigned int mA = A.m(), nA = A.n();       // A: mA x nA
  DynamicSparsityPattern dsp(nA, B.n());           // A^T * B: nA x B.n()

  // Build column adjacency of A (to iterate columns without forming A^T)
  std::vector<std::vector<unsigned int>> col_rows_A(nA);
  for (unsigned int r = 0; r < mA; ++r)
    for (auto it = A.begin(r); it != A.end(r); ++it)
      col_rows_A[it->column()].push_back(r);

  for (unsigned int i = 0; i < nA; ++i) {
    for (unsigned int k : col_rows_A[i]) {          // A(k,i) ≠ 0
      for (auto b = B.begin(k); b != B.end(k); ++b) // B(k,j) ≠ 0
        dsp.add(i, b->column());                    // (A^T B)(i,j) ≠ 0
    }
  }
  std::size_t nnz = 0;
  for (unsigned int r = 0; r < dsp.n_rows(); ++r) {
      nnz += dsp.row_length(r);
  }   
  auto sp = std::make_shared<SparsityPattern>();
  sp->copy_from(dsp);
  return sp;
}