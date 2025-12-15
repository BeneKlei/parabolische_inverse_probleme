#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

#include <MaterialModel.hpp>

#include "ElasticityModel.hpp"


namespace py = pybind11;

PYBIND11_MODULE(elasticity_model, m) {
    // Make sure the base Python class exists/registered first:
    py::module_::import("RBInvParam.problems.shared.material_model");

    // Inherit in the binding:
    py::class_<ElasticityModel, MaterialModel>(m, "ElasticityModel")
        .def(py::init<const MaterialModelConfig&>())
        .def("test", &ElasticityModel::test);
}