#include "MaterialMatricesFactory.hpp"

template class SystemMatrices<3, double>;

template <int dim, typename Number>
void SystemMatrices<dim, Number>::assemble(SparseMatrix<Number> &result, const Vector<Number> &parameters) 
{
    Assert(m_param_space_dim == parameters.size(),
            ExcDimensionMismatch(m_matrices.size(), parameters.size()));

    result = 0;

    unsigned int offset = m_affine ? 1 : 0;
    AssertDimension(parameters.size(), m_matrices.size() - offset);

    if (m_affine)
        result.add(1.0, m_matrices[0]);

    for (unsigned int i = 0; i < parameters.size(); ++i)
        result.add(parameters[i], m_matrices[i + offset]);
}


template <int dim, typename Number>
std::size_t SystemMatrices<dim, Number>::get_param_space_dim() 
{   
    return m_param_space_dim;
}
