#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

#include <MaterialModel.hpp>
#include <MaterialOperatorFactory.hpp>

#include "ElasticityModel.hpp"


namespace py = pybind11;


PYBIND11_MODULE(elasticity_model, m) {
    py::module::import("pymor_dealii_bindings");
    py::class_<ElasticityModel, MaterialModel>(m, "ElasticityModel")
        .def(py::init<const ElasticityModelConfig&>())
        .def("assemble_system_operator", &ElasticityModel::assemble_system_operator)
        .def("setup_system_operator", &ElasticityModel::setup_system_operator)
        .def("set_q", &ElasticityModel::set_q)
        .def_property_readonly(
            "m_op",
            [](ElasticityModel &self) -> BilinearAqOp<double>* {
                return self.m_op.get();
            },
            py::return_value_policy::reference_internal
        );

    py::class_<ElasticityModelConfig, MaterialModelBaseConfig>(m, "ElasticityModelConfig")
         .def(py::init<>())
         .def_readwrite("system_operator_type", &ElasticityModelConfig::system_operator_type)
         .def_readwrite("system_operator_hyperparameter", &ElasticityModelConfig::system_operator_hyperparameter);
}