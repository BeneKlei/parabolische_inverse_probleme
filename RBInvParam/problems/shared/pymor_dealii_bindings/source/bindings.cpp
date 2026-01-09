// This file is part of the pyMOR project (http://www.pymor.org).
// Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

//#include "utils.hpp"
#include "Operators.hpp"
#include "MatrixOperator.hpp"

namespace py = pybind11;

template <typename Number>
void bind_vector(pybind11::module& module) {
  typedef dealii::Vector<Number> Vector;
  typedef typename Vector::size_type size_type;
  py::class_<Vector>(module, "Vector", py::buffer_protocol())
      .def(py::init<>())
      .def(py::init<const Vector&>())
      .def(py::init<const size_type>())
      .def("swap", &Vector::swap)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(py::self * py::self)
      .def(py::self += py::self)
      .def(py::self -= py::self)
      .def(py::self *= Number())
      .def(py::self /= Number())
      .def("axpy", [](Vector& self, Number a, const Vector& x) { self.add(a, x); })
      .def("norm_sqr", &Vector::norm_sqr)
      .def("mean_value", &Vector::mean_value)
      .def("norm_sqr", &Vector::norm_sqr)
      .def("l1_norm", &Vector::l1_norm)
      .def("l2_norm", &Vector::l2_norm)
      .def("lp_norm", &Vector::lp_norm, py::arg("p"))
      .def("linfty_norm", &Vector::linfty_norm)
      .def("all_zero", &Vector::all_zero)
      .def("norm_sqr", &Vector::norm_sqr)
      .def("size", &Vector::size)
      .def("__getitem__",
           [](const Vector& s, size_type i) {
             if (i >= s.size())
               throw py::index_error();
             return s[i];
           })
      .def("__setitem__",
           [](Vector& s, size_type i, Number v) {
             if (i >= s.size())
               throw py::index_error();
             s[i] = v;
           })
      /// Slicing protocol (optional)
      .def("__getitem__",
           [](const Vector& s, py::slice slice) -> Vector* {
             std::size_t start, stop, step, slicelength;
             if (!slice.compute(s.size(), &start, &stop, &step, &slicelength))
               throw py::error_already_set();
             Vector* seq = new Vector(slicelength);
             for (int i = 0; i < slicelength; ++i) {
               (*seq)[i] = s[start];
               start += step;
             }
             return seq;
           })
      .def("__setitem__",
           [](Vector& s, py::slice slice, const Vector& value) {
             std::size_t start, stop, step, slicelength;
             if (!slice.compute(s.size(), &start, &stop, &step, &slicelength))
               throw py::error_already_set();
             if ((size_t)slicelength != value.size())
               throw std::runtime_error("Left and right hand size of slice assignment have different sizes!");
             for (int i = 0; i < slicelength; ++i) {
               s[start] = value[i];
               start += step;
             }
           })
      .def("__setitem__",
           [](Vector& s, py::slice slice, py::array_t<Number> value) {
             std::size_t start, stop, step, slicelength;
             py::buffer_info info = value.request();
             if (!slice.compute(s.size(), &start, &stop, &step, &slicelength))
               throw py::error_already_set();
             if (slicelength != info.size) {
               std::stringstream ss;
               ss << "Left and right hand size of slice assignment have different sizes!";
               ss << slicelength << " vs. " << info.ndim;
               throw std::runtime_error(ss.str());
             }
             for (int i = 0; i < slicelength; ++i) {
               s[start] = *(static_cast<Number*>(info.ptr) + i);
               start += step;
             }
           })
      /// Provide buffer access
      .def_buffer([](Vector& m) -> py::buffer_info {
        return py::buffer_info(&m[0],                                /* Pointer to buffer */
                               sizeof(Number),                       /* Size of one scalar */
                               py::format_descriptor<Number>::value, /* Python struct-style format descriptor */
                               1,                                    /* Number of dimensions */
                               {
                                   m.size(),
                               }, /* Buffer dimensions */
                               {sizeof(Number)});
      })
      .def("__len__", &Vector::size)
      .def("__repr__", [](const Vector& a) {
        std::stringstream ss;
        ss << "<dealii.Vector<Number> with size '" << a.size() << "'>";
        return ss.str();
      });
}

template <typename Number>
void bind_sparse_matrix(pybind11::module& module) {
  using Matrix = dealii::SparseMatrix<Number>;
  using Vector = dealii::Vector<Number>;

  auto cg_solve = [](Matrix& self, Vector& solution, const Vector& rhs) {
    dealii::SolverControl solver_control(20000, 1e-12);
    dealii::SolverCG<> solver(solver_control);
    dealii::PreconditionSSOR<> preconditioner;
    preconditioner.initialize(self, 1.2);
    solver.solve(self, solution, rhs, preconditioner);

    // We have made one addition, though: since we suppress output from the
    // linear solvers, we have to print the number of iterations by hand.
    //std::cout << "   " << solver_control.last_step() << " CG iterations needed to obtain convergence." << std::endl;
  };

  py::class_<Matrix>(module, "SparseMatrix")
      .def(py::init<>())
      .def(py::init<const dealii::SparsityPattern&>())
      .def(py::self *= Number())
      .def("n", [](const Matrix& mat) { return 0 ? mat.empty() : mat.n(); })
      .def("m", [](const Matrix& mat) { return 0 ? mat.empty() : mat.m(); })
      .def("clear", &Matrix::clear)
      .def("l1_norm", &Matrix::l1_norm)
      .def("linfty_norm", &Matrix::linfty_norm)
      //.def("vmult", &Matrix::template vmult<Vector, Vector>)
      .def("vmult",
          [](const Matrix &self, Vector &dst, const Vector &src) {
            py::gil_scoped_release release;
            self.vmult(dst, src);})
      // .def("vmult_batch", [](const Matrix& A, const std::vector<Vector>& src) {
      //     py::gil_scoped_release release;
      //     std::vector<Vector> dst(src.size());
      //     for (size_t i = 0; i < src.size(); ++i)
      //         A.vmult(dst[i], src[i]);
      //     return dst;
      // })
      .def("Tvmult", &Matrix::template Tvmult<Vector, Vector>)
      .def("mmult",
          static_cast<void (Matrix::*)(
              Matrix&, const Matrix&, const Vector&, const bool) const>
          (&Matrix::template mmult<Number, Number>),
          py::arg("C"), 
          py::arg("B"),
          py::arg("V") = Vector(),
          py::arg("rebuild_sparsity_pattern") = true)
      .def("Tmmult",
          static_cast<void (Matrix::*)(
              Matrix&, const Matrix&, const Vector&, const bool) const>
          (&Matrix::template Tmmult<Number, Number>),
          py::arg("C"), 
          py::arg("B"),
          py::arg("V") = Vector(),
          py::arg("rebuild_sparsity_pattern") = true)
      .def("get_sparsity_pattern", &Matrix::get_sparsity_pattern, py::return_value_policy::reference_internal)
      .def("add", (void(Matrix::*)(Number, const Matrix&)) & Matrix::template add<Number>)
      .def("copy_from",
        [](Matrix &self, const Matrix &other) {
            py::gil_scoped_release release;
            return self.copy_from(other);
        })
    .def("reinit",
        [](Matrix &self, const dealii::SparsityPattern &sp) {
            py::gil_scoped_release release;
            self.reinit(sp);
        })
      .def("cg_solve", cg_solve);
}

template <typename Number>
void bind_full_matrix(py::module &module)
{
  using Matrix = dealii::FullMatrix<Number>;
  using Vector = dealii::Vector<Number>;

  py::class_<Matrix>(module, "FullMatrix")
      // constructors
      .def(py::init<>())
      .def(py::init<unsigned int, unsigned int>(), py::arg("m"), py::arg("n"))
      .def("m", &Matrix::m)
      .def("n", &Matrix::n)
      .def("clear", &Matrix::clear)
      .def(py::self *= Number())
      .def("l1_norm", &Matrix::l1_norm)
      .def("linfty_norm", &Matrix::linfty_norm)
      .def("vmult",
          static_cast<void (Matrix::*)(Vector&, const Vector&, bool) const>
          (&Matrix::template vmult<Number>),
          py::arg("dst"), 
          py::arg("src"), 
          py::arg("adding") = false)
      .def("Tvmult",
          static_cast<void(Matrix::*)(Vector&, const Vector&, bool) const>
          (&Matrix::template Tvmult<Number>),
          py::arg("dst"), 
          py::arg("src"), 
          py::arg("adding") = false);
}

template <typename Number>
void bind_ILU_solver(pybind11::module& module) {
  using Matrix = dealii::SparseMatrix<Number>;
  using Vector = dealii::Vector<Number>;
  using SparseILU = dealii::SparseILU<Number>;
  using AdditionalData = typename SparseILU::AdditionalData;

  py::class_<AdditionalData>(module, "AdditionalData")
    .def(py::init<>());

  py::class_<SparseILU>(module, "SparseILU")
    .def(py::init<>())
    .def("initialize",
         static_cast<void (SparseILU::*)(
             const Matrix &,
             const AdditionalData &)>(&SparseILU::initialize),
         py::arg("matrix"),
         py::arg("additional_data") = AdditionalData())
    .def("vmult",
         static_cast<void (SparseILU::*)(
             Vector &,
             const Vector &) const>(&SparseILU::vmult));
}

void bind_sparsity_pattern(pybind11::module& module) {
  py::class_<dealii::SparsityPattern, std::shared_ptr<dealii::SparsityPattern>>(module, "SparsityPattern")
    .def(py::init<>())
    .def("reinit", (void (dealii::SparsityPattern::*)(unsigned int,unsigned int,unsigned int))
                 &dealii::SparsityPattern::reinit)
    .def("n_rows", &dealii::SparsityPattern::n_rows)
    .def("n_cols", &dealii::SparsityPattern::n_cols)
    .def("max_entries_per_row", &dealii::SparsityPattern::max_entries_per_row);
}

template <class Number>
void bind_operators(py::module_& m)
{
  using BaseOperator   = BaseOperator<Number>;
  using MatrixOperator = MatrixOperator<Number>;
  using Vec            = dealii::Vector<Number>;

  py::class_<BaseOperator, std::shared_ptr<BaseOperator>>(m, "BaseOperator")
      .def("apply", &BaseOperator::apply, py::arg("y"), py::arg("x"))
      .def("apply_adjoint", &BaseOperator::apply_adjoint, py::arg("y"), py::arg("x"))
      .def("apply_inverse", &BaseOperator::apply_inverse, py::arg("y"), py::arg("x"))
      .def("apply_inverse_adjoint", &BaseOperator::apply_inverse_adjoint, py::arg("y"), py::arg("x"))
      .def("has_inverse", &BaseOperator::has_inverse)
      .def("has_inverse_adjoint", &BaseOperator::has_inverse_adjoint)
      .def("dim_source", &BaseOperator::dim_source)
      .def("dim_range", &BaseOperator::dim_range);
  
  py::class_<MatrixOperator, BaseOperator, std::shared_ptr<MatrixOperator>>(m, "MatrixOperator")
      .def("apply", &MatrixOperator::apply, py::arg("y"), py::arg("u"))
      .def("apply_adjoint", &MatrixOperator::apply_adjoint, py::arg("y"), py::arg("w"))
      .def("apply_inverse", &MatrixOperator::apply_inverse, py::arg("y"), py::arg("f"))
      .def("apply_inverse_adjoint", &MatrixOperator::apply_inverse_adjoint, py::arg("y"), py::arg("f"))

      .def("has_inverse", &MatrixOperator::has_inverse)
      .def("has_inverse_adjoint", &MatrixOperator::has_inverse_adjoint)

      .def("dim_source", &MatrixOperator::dim_source)
      .def("dim_range", &MatrixOperator::dim_range);
}

PYBIND11_MODULE(pymor_dealii_bindings, m) {
  m.doc() = "Python bindings for deal.II";
  bind_sparsity_pattern(m);
  bind_vector<double>(m);
  bind_full_matrix<double>(m);
  bind_sparse_matrix<double>(m);
  bind_operators<double>(m);
  //bind_ILU_solver<double>(m);
  //bind_cgsolver<double>(m);

  // auto utils = m.def_submodule("utils");
  // utils.def("make_product_sparsity_AB", &make_product_sparsity_AB);
  // utils.def("make_product_sparsity_ATB", &make_product_sparsity_ATB);

}
