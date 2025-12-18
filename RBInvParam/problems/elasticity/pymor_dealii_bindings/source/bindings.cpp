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

#include <deal.II/base/timer.h>

#include "utils.hpp"

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

      // .def("vmult",
      //     (void(Matrix::*)(Vector&, const Vector&, bool) const) & Matrix::template vmult<Number>
      // )
      // .def("Tvmult",
      //     (void(Matrix::*)(Vector&, const Vector&, bool) const) & Matrix::template Tvmult<Number>
      // );
      


      // // element access helpers
      // .def("set",
      //      [](Matrix &A, unsigned int i, unsigned int j, Number v) { A(i, j) = v; })
      // .def("get",
      //      [](const Matrix &A, unsigned int i, unsigned int j) { return A(i, j); })
      // .def("add_to_entry",
      //      [](Matrix &A, unsigned int i, unsigned int j, Number v) { A(i, j) += v; })

      // // vector–matrix products (keep templates explicit like your sparse binding)
      // // matrix–matrix products (in-place, 9.6.0 signatures)
      // .def("mmult",
      //      (void (Matrix::*)(Matrix &, const Matrix &, const bool) const)
      //          & Matrix::mmult,
      //      py::arg("C"), py::arg("B"), py::arg("add") = false)
      // .def("Tmmult",
      //      (void (Matrix::*)(Matrix &, const Matrix &, const bool) const)
      //          & Matrix::Tmmult,
      //      py::arg("C"), py::arg("B"), py::arg("add") = false)

      // // BLAS-like ops
      // .def("add",
      //      (void (Matrix::*)(Number, const Matrix &)) & Matrix::template add<Number>,
      //      py::arg("a"), py::arg("A"))

      // // convenience: set whole row/column from a dealii::Vector
      // .def("set_row",
      //      [](Matrix &A, unsigned int i, const Vector &row) {
      //        if (row.size() != A.n())
      //          throw std::runtime_error("set_row: size mismatch");
      //        for (unsigned int j = 0; j < A.n(); ++j) A(i, j) = row[j];
      //      },
      //      py::arg("i"), py::arg("row"))
      // .def("set_column",
      //      [](Matrix &A, unsigned int j, const Vector &col) {
      //        if (col.size() != A.m())
      //          throw std::runtime_error("set_column: size mismatch");
      //        for (unsigned int i = 0; i < A.m(); ++i) A(i, j) = col[i];
      //      },
      //      py::arg("j"), py::arg("col"))

      // // convenience: zero everything (handy “reset”)
      // .def("set_to_zero", [](Matrix &A) { A = 0; });
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

// template <typename Number>
// void bind_cgsolver(pybind11::module& module) {
//   using Matrix = dealii::SparseMatrix<Number>;
//   using Vector = dealii::Vector<Number>;

//   class CGSolverWrapper
//   {
//   public:
//     CGSolverWrapper(Matrix &A,
//                     const double tol = 1e-12,
//                     const unsigned int max_steps = 20000,
//                     const double ssor_omega = 1.2)
//         : matrix(A),
//           solver_control(max_steps, tol),
//           solver(solver_control)
//     {
//       preconditioner.initialize(matrix, ssor_omega);
//     }

//     void solve(Vector &solution, const Vector &rhs)
//     {
//       solver.solve(matrix, solution, rhs, preconditioner);
//       std::cout << "   " << solver_control.last_step()
//                 << " CG iterations needed to obtain convergence."
//                 << std::endl;
//     }

//   private:
//     Matrix &matrix;
//     dealii::SolverControl solver_control;
//     dealii::SolverCG<> solver;
//     dealii::PreconditionSSOR<> preconditioner;
//   };

//   py::class_<CGSolverWrapper>(module, "CGSolver")
//     .def(py::init<Matrix &, double, unsigned int, double>(),
//           py::arg("matrix"),
//           py::arg("tol") = 1e-12,
//           py::arg("max_steps") = 20000,
//           py::arg("ssor_omega") = 1.2)
//     .def("solve", &CGSolverWrapper::solve);

// }

template <typename Number>
class CGSolverWrapper
{
public:
  using Matrix = dealii::SparseMatrix<Number>;
  using Vector = dealii::Vector<Number>;

  CGSolverWrapper(Matrix &A_,
                  double tol = 1e-12,
                  unsigned int max_steps = 20000,
                  double ssor_omega = 1.2)
    : A(&A_)
    , tol_(tol)
    , max_steps_(max_steps)
    , omega_(ssor_omega)
  {
    rebuild_preconditioner();
  }

  void rebuild_preconditioner()
  {
    // (Re)build cached preconditioner from current matrix values
    prec.initialize(*A, omega_);
    prec_ready = true;
  }

  void set_ssor_omega(double omega)
  {
    omega_ = omega;
    prec_ready = false; // force rebuild on next solve
  }


  void solve(Vector &x, const Vector &b)
  {
    if (!prec_ready)
      rebuild_preconditioner();

    dealii::SolverControl control(max_steps_, tol_);
    dealii::SolverCG<> solver(control);

    double cg_time = 0.0;

    {
      dealii::Timer timer;
      timer.start();

      solver.solve(*A, x, b, prec);

      timer.stop();
      cg_time = timer.wall_time();
    }

    std::cout << "CG iterations: " << control.last_step()
              << ", CG time: " << cg_time << " s"
              << ", final residual: " << control.last_value()
              << std::endl;
  }

private:
  Matrix *A; // non-owning; pybind keep_alive ensures lifetime
  double tol_;
  unsigned int max_steps_;
  double omega_;
  bool prec_ready = false;

  dealii::PreconditionSSOR<Matrix> prec;
};

template <typename Number>
void bind_cgsolver(pybind11::module& m)
{
  namespace py = pybind11;
  using Matrix = dealii::SparseMatrix<Number>;
  using Vector = dealii::Vector<Number>;
  using CGS    = CGSolverWrapper<Number>;

  py::class_<CGS>(m, "CGSolver")
    .def(py::init<Matrix&, double, unsigned int, double>(),
         py::arg("matrix"),
         py::arg("tol") = 1e-12,
         py::arg("max_steps") = 20000,
         py::arg("ssor_omega") = 1.2,
         py::keep_alive<1, 2>()   // solver keeps matrix alive
    )
    .def("solve", [](CGS &self, Vector &x, const Vector &b){
        py::gil_scoped_release r;
        self.solve(x, b);
    })
    .def("rebuild_preconditioner", [](CGS &self){
        py::gil_scoped_release r;
        self.rebuild_preconditioner();
    })
    .def("set_ssor_omega", &CGS::set_ssor_omega);
}




PYBIND11_MODULE(pymor_dealii_bindings, m) {
  m.doc() = "Python bindings for deal.II";
  bind_sparsity_pattern(m);
  bind_vector<double>(m);
  bind_full_matrix<double>(m);
  bind_sparse_matrix<double>(m);
  bind_cgsolver<double>(m);
  //bind_ILU_solver<double>(m);
  //bind_cgsolver<double>(m);

  // auto utils = m.def_submodule("utils");
  // utils.def("make_product_sparsity_AB", &make_product_sparsity_AB);
  // utils.def("make_product_sparsity_ATB", &make_product_sparsity_ATB);

}
