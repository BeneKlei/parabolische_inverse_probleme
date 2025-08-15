#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

using namespace dealii;

typedef double Number;

class MatrixStack 
{
public:
    explicit MatrixStack() = default;
    void sum(SparseMatrix<Number>& result, Vector<Number>& weights);
    void resize(uint32_t size);
    void reinit(const SparsityPattern& sparsity_pattern);

    const SparseMatrix<Number>& get_matrix(size_t index);
    size_t get_size();

    // TODO Move protected. Allow the return of iterator
    std::vector<SparseMatrix<Number>> m_matrices;
    SparseMatrix<Number> m_sum;

//protected:
    //std::vector<SparseMatrix<Number>> m_matrices;
};
