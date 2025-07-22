#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

using namespace dealii;

typedef double Number;
typedef SparseMatrix<Number> Matrix;
//typedef std::vector<Matrix> Matrices;

class MatrixStack 
{
public:
    explicit MatrixStack() = default;
    void sum(SparseMatrix<Number>& result, Vector<Number>& weights);
    uint32_t length() {return m_matrices.size();};
    void resize(uint32_t size);
    void reinit(const SparsityPattern& sparsity_pattern);

    //void push_matrix(Matrix& matrix);

    std::vector<Matrix> m_matrices;

  //const dealii::SparseMatrix<Number>& sum(std::vector<Number>& weights);
    SparseMatrix<Number> m_sum;
// private:
//     SparseMatrix<Number> m_sum;
 
};
