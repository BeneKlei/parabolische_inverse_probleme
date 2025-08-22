#include "MaterialMatricesFactory.hpp"

template class SystemMatrices<3, double>;

template <int dim, typename Number>
void SystemMatrices<dim, Number>::assemble(SparseMatrix<Number> &result, const Vector<Number> &parameters) 
{
    Assert(m_param_space_dim == parameters.size(),
           ExcDimensionMismatch(m_matrices.size(), parameters.size()));

    // Reset result before accumulating
    //std::size_t start_idx;

    if (m_affine) {
        result = m_matrices[0];
    } else {
        result = 0;
        result.add(parameters[0], m_matrices[0]);
    }
    
    for (unsigned int i = 1; i < m_matrices.size(); ++i) {
        result.add(parameters[i], m_matrices[i]);
    }
  }