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
#include "ROMProjector.hpp"

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

template <typename Number>
void bind_operators(py::module_& m)
{
  using BaseOp   = BaseOperator<Number>;
  using SparseOp = SparseMatrixOperator<Number>;
  using FullOp   = FullMatrixOperator<Number>;

  auto bind_base_methods = [](auto& cls) -> auto& {
    return cls
      .def("apply", &BaseOp::apply, py::arg("y"), py::arg("x"))
      .def("apply_adjoint", &BaseOp::apply_adjoint, py::arg("y"), py::arg("x"))
      .def("apply_inverse",
           &BaseOp::apply_inverse,
           py::arg("y"),
           py::arg("x"),
           py::arg("rtol") = 0.0,
           py::arg("atol") = 1e-12,
           py::arg("maxiter") = 20000)
      .def("apply_inverse_adjoint",
           &BaseOp::apply_inverse_adjoint,
           py::arg("y"),
           py::arg("x"),
           py::arg("rtol") = 0.0,
           py::arg("atol") = 1e-12,
           py::arg("maxiter") = 20000)
      .def("jacobian", &BaseOp::jacobian, py::arg("u"))
      .def("dim_source", &BaseOp::dim_source)
      .def("dim_range", &BaseOp::dim_range)
      .def_readonly("linear", &BaseOp::m_linear);
  };

  auto base_cls =
    py::class_<BaseOp, std::unique_ptr<BaseOp>>(m, "BaseOperator");

  bind_base_methods(base_cls);

  py::class_<SparseOp, BaseOp, std::unique_ptr<SparseOp>>(m, "SparseMatrixOperator")
      .def(py::init<typename SparseOp::MatV&>(), py::arg("matrix"))
      .def("get_matrix", &SparseOp::get_matrix, py::return_value_policy::reference_internal);

  py::class_<FullOp, BaseOp, std::unique_ptr<FullOp>>(m, "FullMatrixOperator")
      .def(py::init<typename FullOp::MatV&>(), py::arg("matrix"))
      .def("get_matrix", &FullOp::get_matrix, py::return_value_policy::reference_internal);
}


// template <typename Number>
// void bind_operators(py::module_& m)
// {
//   using BaseOp   = BaseOperator<Number>;
//   using SparseOp = SparseMatrixOperator<Number>;
//   using FullOp   = FullMatrixOperator<Number>;
//   using Vec      = dealii::Vector<Number>;

//   py::class_<BaseOp, std::unique_ptr<BaseOp>>(m, "BaseOperator")
//       .def("apply", &BaseOp::apply, py::arg("y"), py::arg("x"))
//       .def("apply_adjoint", &BaseOp::apply_adjoint, py::arg("y"), py::arg("x"))
//       .def("apply_inverse", &BaseOp::apply_inverse, py::arg("y"), py::arg("x"))
//       .def("apply_inverse_adjoint", &BaseOp::apply_inverse_adjoint, py::arg("y"), py::arg("x"))
//       .def("jacobian", &BaseOp::jacobian, py::arg("u"))
//       .def("dim_source", &BaseOp::dim_source)
//       .def("dim_range", &BaseOp::dim_range)
//       .def_readonly("linear", &BaseOp::m_linear);

//   py::class_<SparseOp, BaseOp, std::unique_ptr<SparseOp>>(m, "SparseMatrixOperator")
//       .def(py::init<typename SparseOp::MatV&>(), py::arg("matrix"))
//       .def("apply", &SparseOp::apply, py::arg("y"), py::arg("u"))
//       .def("apply_adjoint", &SparseOp::apply_adjoint, py::arg("y"), py::arg("w"))
//       .def("apply_inverse", &SparseOp::apply_inverse, py::arg("y"), py::arg("f"))
//       .def("apply_inverse_adjoint", &SparseOp::apply_inverse_adjoint, py::arg("y"), py::arg("f"))
//       .def("jacobian", &SparseOp::jacobian, py::arg("u"))
//       .def("get_matrix", &SparseOp::get_matrix, py::return_value_policy::reference_internal)
//       .def("dim_source", &SparseOp::dim_source)
//       .def("dim_range", &SparseOp::dim_range)
//       .def_readonly("linear", &BaseOp::m_linear);

//   py::class_<FullOp, BaseOp, std::unique_ptr<FullOp>>(m, "FullMatrixOperator")
//       .def(py::init<typename FullOp::MatV&>(), py::arg("matrix"))
//       .def("apply", &FullOp::apply, py::arg("y"), py::arg("u"))
//       .def("apply_adjoint", &FullOp::apply_adjoint, py::arg("y"), py::arg("w"))
//       .def("apply_inverse", &FullOp::apply_inverse, py::arg("y"), py::arg("f"))
//       .def("apply_inverse_adjoint", &FullOp::apply_inverse_adjoint, py::arg("y"), py::arg("f"))
//       .def("jacobian", &FullOp::jacobian, py::arg("u"))
//       .def("get_matrix", &FullOp::get_matrix, py::return_value_policy::reference_internal)
//       .def("dim_source", &FullOp::dim_source)
//       .def("dim_range", &FullOp::dim_range)
//       .def_readonly("linear", &BaseOp::m_linear);
// }

// bindings_rom_projector.cpp
//
// Drop this into your existing bindings.cpp (or compile as a separate TU and link)
// and call bind_rom_projector<double>(m) from your module init.
//
// This version is "clean":
// - It does NOT take py::object operators.
// - It takes the *native C++* pd2::SparseMatrixOperator<Number> objects
//   (the ones you already expose via pymor_dealii_bindings).
// - It extracts &op.get_matrix() and stores SparseMatrix* pointers inside ReducedProjector.
//
// Python usage then becomes:
//   proj.set_operators([A.op for A in python_sparse_matrix_operator_wrappers])
// because A.op is the native pd2 operator.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include "ROMProjector.hpp"

// IMPORTANT: include the header that declares your C++ SparseMatrixOperator template.
// From your snippet, this is likely Operators.hpp or the header where
// `template<class Number> using SparseMatrixOperator = MatrixOperator<Number, SparseMatrix<Number>>;`
#include "Operators.hpp"   // <-- adjust to your actual include path

namespace py = pybind11;

template <class Number>
void bind_rom_projector(py::module_ &m)
{
  using Basis = DenseBasis<Number>;
  using Proj  = ReducedProjector<Number>;

  // This is your *C++* deal.II-backed operator type (NOT the Python wrapper class)
  using CppSparseOp = SparseMatrixOperator<Number>;
  using Mat         = dealii::SparseMatrix<Number>;

  py::class_<Basis, std::shared_ptr<Basis>>(m, "DenseBasis")
    .def(py::init<int,int>(), py::arg("n_full"), py::arg("capacity"))
    .def_property_readonly("n", &Basis::n)
    .def_property_readonly("r", &Basis::r)
    .def_property_readonly("capacity", &Basis::capacity)
    .def("reserve", &Basis::reserve, py::arg("new_cap"))

    // basis.append(W) where W is numpy (n×k)
    .def("append", [](Basis &self, py::array_t<Number, py::array::c_style | py::array::forcecast> W) {
        if (W.ndim() != 2)
          throw std::runtime_error("DenseBasis.append: W must be 2D (n×k)");
        const int n = static_cast<int>(W.shape(0));
        const int k = static_cast<int>(W.shape(1));
        const int s0 = static_cast<int>(W.strides(0) / (py::ssize_t)sizeof(Number));
        const int s1 = static_cast<int>(W.strides(1) / (py::ssize_t)sizeof(Number));
        self.append_from_numpy_ptr((const Number*)W.data(), n, k, s0, s1);
      }, py::arg("W"))

    // basis.remove([idxs])
    .def("remove", [](Basis &self, py::array_t<int, py::array::c_style | py::array::forcecast> idxs) {
        if (idxs.ndim() != 1)
          throw std::runtime_error("DenseBasis.remove: idxs must be 1D");
        self.remove_swap((const int*)idxs.data(), static_cast<int>(idxs.shape(0)));
      }, py::arg("idxs"))
    ;

  py::class_<Proj>(m, "ReducedProjector")
    .def(py::init<std::shared_ptr<Basis>, int, bool>(),
         py::arg("basis"), py::arg("max_r"), py::arg("symmetric")=true)

    .def_property_readonly("n", &Proj::n)
    .def_property_readonly("r", &Proj::r)
    .def_property_readonly("num_operators", &Proj::num_operators)

    .def("notify_basis_changed", &Proj::notify_basis_changed)

    // CLEAN: take a list of *native C++* SparseMatrixOperator<Number> objects
    //
    // Python will pass: [A.op for A in python_wrapper_ops]
    // where A.op is pd2.SparseMatrixOperator (C++ object bound by pybind).
    //
    .def("set_operators", [](Proj &self, const std::vector<const CppSparseOp*> &ops) {
      std::vector<const Mat*> mats;
      mats.reserve(ops.size());
      for (auto *op : ops) {
        if (!op) throw std::runtime_error("null operator");
        mats.push_back(&op->get_matrix());
      }
      self.set_operators(mats);
    }, py::arg("ops"))

    .def("project_full", &Proj::project_full, py::arg("q"))
    .def("project_full_all", &Proj::project_full_all)
    .def("update_after_append", &Proj::update_after_append, py::arg("k"))

    // Return numpy (r×r)
    .def("get", [](Proj &self, int q) {
        const int r = self.r();
        py::array_t<Number> out({r, r});
        auto o = out.template mutable_unchecked<2>();

        Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic> tmp(r, r);
        self.get(q, tmp);

        for (int i = 0; i < r; ++i)
          for (int j = 0; j < r; ++j)
            o(i, j) = tmp(i, j);

        return out;
      }, py::arg("q"))
    ;
}


PYBIND11_MODULE(pymor_dealii_bindings, m) {
  m.doc() = "Python bindings for deal.II";
  bind_sparsity_pattern(m);
  bind_vector<double>(m);
  bind_full_matrix<double>(m);
  bind_sparse_matrix<double>(m);
  bind_operators<double>(m);
  bind_rom_projector<double>(m);
  //bind_ILU_solver<double>(m);
  //bind_cgsolver<double>(m);

  // auto utils = m.def_submodule("utils");
  // utils.def("make_product_sparsity_AB", &make_product_sparsity_AB);
  // utils.def("make_product_sparsity_ATB", &make_product_sparsity_ATB);

}
