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
      .def("setup_material_operator", &ElasticityModel::setup_material_operator)

      .def("assemble_A_q", 
         &ElasticityModel::assemble_A_q, 
         py::arg("q_np"),
         py::arg("linear_part_only") = false
      )

      .def("assemble_partial_q_A_q_u", 
         &ElasticityModel::assemble_partial_q_A_q_u, 
         py::arg("u")
      )

      .def("assemble_partial_u_A_q_u", 
         &ElasticityModel::assemble_partial_u_A_q_u, 
         py::arg("q_np")
      )

      .def("get_translation_operator", 
         &ElasticityModel::get_translation_operator
      )
      
      .def_readonly("m_has_translation_operator", &ElasticityModel::m_has_translation_operator);

    py::class_<ElasticityModelConfig, MaterialModelBaseConfig>(m, "ElasticityModelConfig")
         .def(py::init<>())
         .def_readwrite("material_operator_type", &ElasticityModelConfig::material_operator_type)
         .def_readwrite("material_operator_hyperparameter", &ElasticityModelConfig::material_operator_hyperparameter);
}