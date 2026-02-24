#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/grid/tria.h>

#include <cmath>

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

    /**
   * Cheap structural checks (counts/levels).
   */
    template <int dim, class TriangulationType>
    bool same_tria_structure(const TriangulationType &a, const TriangulationType &b);

    /**
     * Compare vertex coordinates + active-cell connectivity (vertex indices per cell).
     *
     * Assumes both triangulations have identical vertex ordering and active-cell ordering.
     * If ordering can differ, use a signature-based comparison instead.
     */
    template <int dim, class TriangulationType>
    bool same_tria_geometry_and_connectivity(const TriangulationType &a,
                                             const TriangulationType &b,
                                             double tol = 0.0);

    // Quadrature equivalence: same points+weights (within tol)
    template <int dim, typename Number = double>
    bool same_quadrature(const dealii::Quadrature<dim> &a,
                        const dealii::Quadrature<dim> &b,
                        Number tol = Number(0));

    // Mapping equivalence: same dynamic type; optionally compare degree for MappingQ
    template <int dim>
    bool same_mapping_configuration(const dealii::Mapping<dim> &a,
                                    const dealii::Mapping<dim> &b);

    SparsityPattern make_product_sparsity_AB(const SparseMatrix<double>& A, const SparseMatrix<double>& B);
    SparsityPattern make_product_sparsity_ATB(const SparseMatrix<double>& A, const SparseMatrix<double>& B);

}