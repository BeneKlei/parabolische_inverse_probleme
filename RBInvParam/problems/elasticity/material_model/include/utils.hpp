
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

using namespace dealii;

namespace utils 
{
SparsityPattern make_product_sparsity_AB(const SparseMatrix<double>& A, const SparseMatrix<double>& B);
SparsityPattern make_product_sparsity_ATB(const SparseMatrix<double>& A, const SparseMatrix<double>& B);
}