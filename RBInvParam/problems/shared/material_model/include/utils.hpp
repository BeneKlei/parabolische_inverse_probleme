#pragma once

#include <deal.II/base/point.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

using namespace dealii;

namespace utils 
{
template <int dim>
Point<dim> vec_to_point(const std::vector<double> &coords)
{
    AssertDimension(coords.size(), dim);

    Point<dim> p;
    for (unsigned int d = 0; d < dim; ++d)
        p[d] = coords[d];
    return p;
}

SparsityPattern make_product_sparsity_AB(const SparseMatrix<double>& A, const SparseMatrix<double>& B);
SparsityPattern make_product_sparsity_ATB(const SparseMatrix<double>& A, const SparseMatrix<double>& B);

}