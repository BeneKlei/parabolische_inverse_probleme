#include "utils.hpp"

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <cmath>
#include <typeinfo>

SparsityPattern utils::make_product_sparsity_AB(const SparseMatrix<double>& A,
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
    SparsityPattern sp = SparsityPattern();
    sp.copy_from(dsp);
    sp.compress();
    return sp;
}

SparsityPattern utils::make_product_sparsity_ATB(const SparseMatrix<double>& A,
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
    SparsityPattern sp = SparsityPattern();
    sp.copy_from(dsp);
    sp.compress();
    return sp;
}

template <int dim, class TriangulationType>
bool utils::same_tria_structure(const TriangulationType &a, const TriangulationType &b)
{
    if (a.n_active_cells() != b.n_active_cells()) return false;
    if (a.n_cells()        != b.n_cells())        return false;
    if (a.n_vertices()     != b.n_vertices())     return false;
    if (a.n_levels()       != b.n_levels())       return false;
    return true;
}

template <int dim, class TriangulationType>
bool utils::same_tria_geometry_and_connectivity(const TriangulationType &a,
                                                const TriangulationType &b,
                                                const double tol)
{
    if (!same_tria_structure<dim>(a, b))
      return false;

    // deal.II 9.6: get_vertices() returns a std::vector<Point<dim>>& (or const&)
    const auto &va = a.get_vertices();
    const auto &vb = b.get_vertices();

    // Vertex coordinates (assumes same vertex numbering)
    for (unsigned int v = 0; v < a.n_vertices(); ++v)
      for (unsigned int d = 0; d < dim; ++d)
        if (std::abs(va[v][d] - vb[v][d]) > tol)
          return false;

    // Active-cell connectivity (assumes same active-cell iteration order)
    auto ca = a.begin_active();
    auto cb = b.begin_active();
    for (; ca != a.end(); ++ca, ++cb)
      for (unsigned int i = 0; i < dealii::GeometryInfo<dim>::vertices_per_cell; ++i)
        if (ca->vertex_index(i) != cb->vertex_index(i))
          return false;

    return true;
}

template <int dim, typename Number>
bool utils::same_quadrature(const dealii::Quadrature<dim> &a,
                            const dealii::Quadrature<dim> &b,
                            const Number tol)
{
    if (a.size() != b.size())
        return false;

    // deal.II Quadrature exposes points/weights
    const auto &pa = a.get_points();
    const auto &pb = b.get_points();

    for (unsigned int i = 0; i < a.size(); ++i)
    {
        // points
        for (unsigned int d = 0; d < dim; ++d)
        if (std::abs(pa[i][d] - pb[i][d]) > tol)
            return false;

        // weights
        if (std::abs(a.weight(i) - b.weight(i)) > tol)
        return false;
    }

    return true;
}

template <int dim>
bool utils::same_mapping_configuration(const dealii::Mapping<dim> &a,
                                       const dealii::Mapping<dim> &b)
{
    // Baseline: same dynamic type
    if (typeid(a) != typeid(b))
        return false;

    // MappingQ: compare polynomial degree
    if (const auto *aq = dynamic_cast<const dealii::MappingQ<dim> *>(&a))
    {
        const auto *bq = dynamic_cast<const dealii::MappingQ<dim> *>(&b);
        return (bq != nullptr) && (aq->get_degree() == bq->get_degree());
    }

    // MappingQGeneric: compare degree as well (if you use it)
    if (const auto *aqg = dynamic_cast<const dealii::MappingQGeneric<dim> *>(&a))
    {
        const auto *bqg = dynamic_cast<const dealii::MappingQGeneric<dim> *>(&b);
        return (bqg != nullptr) && (aqg->get_degree() == bqg->get_degree());
    }

    // For other mapping types: same type is the best portable equivalence check.
    return true;
}



// ---- Explicit instantiations for what your symbol shows: dim=3, Triangulation<3,3> ----
template bool utils::same_tria_structure<3, dealii::Triangulation<3,3>>(
  const dealii::Triangulation<3,3>&, const dealii::Triangulation<3,3>&);

template bool utils::same_tria_geometry_and_connectivity<3, dealii::Triangulation<3,3>>(
  const dealii::Triangulation<3,3>&, const dealii::Triangulation<3,3>&, double);

template bool utils::same_quadrature<3, double>(
    const dealii::Quadrature<3>&, const dealii::Quadrature<3>&, double);

template bool utils::same_mapping_configuration<3>(
    const dealii::Mapping<3>&, const dealii::Mapping<3>&);