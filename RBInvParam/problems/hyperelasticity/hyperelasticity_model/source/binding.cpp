#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

#include <MaterialModel.hpp>
#include <MaterialOperatorFactory.hpp>

#include "HyperElasticityModel.hpp"
#include "StoredEnergyFunction.hpp"
#include "StoredEnergyOperator.hpp"

namespace py = pybind11;

template <typename Number>
void bind_operator(py::module_& m)
{
    using BaseOp   = BaseOperator<Number>;
    using SpMatOp  = SparseMatrixOperator<Number>;

    using StorEneOp      = StoredEnergyOperator<3, Number>;
    using StorEneJacOp   = StoredEnergyJacobianOperator<3, Number>;
    using StorEneParOp   = StoredEnergyParamDerivOperator<3, Number>;

    // --- StoredEnergyOperator ---
    py::class_<StorEneOp, BaseOp, std::unique_ptr<StorEneOp>>(m, "StoredEnergyOperator")
        .def("apply", &StorEneOp::apply, py::arg("y"), py::arg("x"))
        .def("dim_source", &StorEneOp::dim_source)
        .def("dim_range", &StorEneOp::dim_range)
        .def("jacobian", &StorEneOp::jacobian)
        .def_readonly("linear", &BaseOp::m_linear);

    // --- StoredEnergyJacobianOperator ---    
    py::class_<StorEneJacOp, SpMatOp, std::unique_ptr<StorEneJacOp>>(m, "StoredEnergyJacobianOperator")
        .def("apply", &SpMatOp::apply, py::arg("y"), py::arg("x"))
        .def("dim_source", &SpMatOp::dim_source)
        .def("dim_range", &SpMatOp::dim_range)
        .def_readonly("linear", &BaseOp::m_linear);

    // --- StoredEnergyParamDerivOperator ---
    py::class_<StorEneParOp, BaseOp, std::unique_ptr<StorEneParOp>>(m, "StoredEnergyParamDerivOperator")
        .def("apply", &StorEneParOp::apply, py::arg("y"), py::arg("d"))
        .def("dim_source", &StorEneParOp::dim_source)
        .def("dim_range", &StorEneParOp::dim_range)
        .def_readonly("linear", &BaseOp::m_linear);
}

PYBIND11_MODULE(hyperelasticity_model, m) {
    py::module::import("pymor_dealii_bindings");

    bind_operator<double>(m);

    py::class_<HyperElasticityModel, MaterialModel>(m, "HyperElasticityModel")
     .def(py::init<const HyperElasticityModelConfig&>())
     .def("setup_material_operator", &HyperElasticityModel::setup_material_operator)

     .def("assemble_A_q", 
          &HyperElasticityModel::assemble_A_q, 
          py::arg("q_np") = py::none()
     )

     // .def("assemble_partial_u_A_q_u", 
     //      &HyperElasticityModel::assemble_partial_u_A_q_u, 
     //      py::arg("q_np"),
     //      py::arg("u")
     // )

     .def("assemble_partial_q_A_q_u", 
          &HyperElasticityModel::assemble_partial_q_A_q_u, 
          py::arg("q_np"),
          py::arg("u")
     );

    //   .def("get_translation_operator", 
    //      &ElasticityModel::get_translation_operator
    //   )
      
    //   .def_readonly("m_has_translation_operator", &ElasticityModel::m_has_translation_operator);

    py::class_<HyperElasticityModelConfig, MaterialModelBaseConfig>(m, "HyperElasticityModelConfig")
         .def(py::init<>())
         .def_readwrite("se_type", &HyperElasticityModelConfig::se_type)
         .def_readwrite("se_hyperparameter", &HyperElasticityModelConfig::se_hyperparameter);

    py::enum_<StoredEnergyFunctionType>(m, "StoredEnergyFunctionType")
         .value("NeoHookean", StoredEnergyFunctionType::NeoHookean)
         .value("Hookean", StoredEnergyFunctionType::Hookean)
         .export_values();
     

}
