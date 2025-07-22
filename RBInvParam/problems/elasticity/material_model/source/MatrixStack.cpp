#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <cassert>

#include "MatrixStack.hpp"

void MatrixStack::resize(uint32_t size) 
{
    if (m_matrices.size() <= size) {
        m_matrices.resize(size);
    } else {
        throw std::runtime_error("MatrixStack::resize(): Cannot shrink matrix stack"); 
    }
}

void MatrixStack::reinit(const SparsityPattern& sparsity_pattern)
{
    if (m_matrices.empty())
        throw std::runtime_error("MatrixStack::reinit(): empty matrix stack");

    for (unsigned int i = 0; i < m_matrices.size(); i++)
    {
        m_matrices[i].reinit(sparsity_pattern);
        // std::cout << "Number of non-zero entries: " << m_matrices[i].n_nonzero_elements() << std::endl;

        // std::cout << "Matrix memory usage: "
        //   << dealii::MemoryConsumption::memory_consumption(m_matrices[i])
        //   << " bytes" << std::endl;
    }
}

void MatrixStack::sum(SparseMatrix<Number>& result, Vector<Number>& weights) 
{
    assert(m_matrices.size() == weights.size());
    for (unsigned int i = 0; i < m_matrices.size(); i++) {
        result.add(weights[i], m_matrices[i]);
    }
}