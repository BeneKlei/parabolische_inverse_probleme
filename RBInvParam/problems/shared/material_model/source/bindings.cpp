// bindings.cpp (refactored: no `using namespace dealii;`, cleaned includes, safer policies)
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "MaterialModel.hpp"

// Optional: only keep if you actually reference these types directly here.
// Otherwise, remove to reduce compile time.
// #include "BodyForceFactory.hpp"
// #include "BoundaryConditionFactory.hpp"
// #include "ObservationOperatorFactory.hpp"
// #include "ProductFactory.hpp"
// #include "ObservationSpaceProductFactory.hpp"
#include "MaterialOperatorFactory.hpp"

namespace py = pybind11;

PYBIND11_MODULE(material_model, m)
{
  // Ensure the deal.II operator wrappers exist before exposing your module types
  py::module_::import("pymor_dealii_bindings");

  // -------------------------- MaterialModel --------------------------

  py::class_<MaterialModel>(m, "MaterialModel")
      // .def(py::init<const MaterialModelBaseConfig&>())  // enable when you have concrete subclasses
      .def("setup_system", &MaterialModel::setup_system)

      // scalar/public fields
      .def_readonly("delta_t", &MaterialModel::delta_t)
      .def_readonly("param_space_dim", &MaterialModel::m_param_dim)
      .def_readonly("state_space_dim", &MaterialModel::m_state_dim)
      .def_readonly("observation_space_dim", &MaterialModel::m_observation_space_dim)
      .def_readonly("has_translation_operator", &MaterialModel::m_has_translation_operator)

      // assembly API returning unique_ptr<SparseMatrixOperator<Number>>
      .def("assemble_param_product_op", &MaterialModel::assemble_param_product_op)
      .def("assemble_state_product_op", &MaterialModel::assemble_state_product_op)
      .def("assemble_product_C_op", &MaterialModel::assemble_product_C_op)
      .def("assemble_mass_op", &MaterialModel::assemble_mass_op)
      .def("assemble_observation_op", &MaterialModel::assemble_observation_op)
      .def("assemble_bilinear_cost_op", &MaterialModel::assemble_bilinear_cost_op)

      // data
      .def_readonly("force_list", &MaterialModel::m_force_list)

      // utilities
      // NOTE: these are void functions, return_value_policy is unnecessary; keep default.
      .def("clear_rhs_boundary_dofs", &MaterialModel::clear_rhs_boundary_dofs)
      .def("save_state", &MaterialModel::save_state)
      .def("save_time_series", &MaterialModel::save_time_series)

      // contexts (references into MaterialModel) -> must use reference_internal
      .def("state_space_context",
           &MaterialModel::state_space_context,
           py::return_value_policy::reference_internal)
      .def("param_space_context",
           &MaterialModel::param_space_context,
           py::return_value_policy::reference_internal)

      // flags (returning const bool&) -> reference_internal is OK, but returning by value is simpler.
      // If you change the C++ signature to `bool q_time_dep() const`, drop policies.
      .def("q_time_dep",
           &MaterialModel::q_time_dep,
           py::return_value_policy::reference_internal)
      .def("A_affine",
           &MaterialModel::A_affine,
           py::return_value_policy::reference_internal)
      .def("A_q_linear",
           &MaterialModel::A_q_linear,
           py::return_value_policy::reference_internal);

  // -------------------------- Enums --------------------------

  py::enum_<MaterialOperatorType>(m, "MaterialOperatorType")
      .value("Cosserat", MaterialOperatorType::Cosserat)
      .value("CosseratDelamination", MaterialOperatorType::CosseratDelamination)
      .value("CosseratSpatial", MaterialOperatorType::CosseratSpatial)
      .export_values();

  py::enum_<BodyForceType>(m, "BodyForceType")
      .value("CenterExcite", BodyForceType::CenterExcite)
      .value("Gaussian", BodyForceType::Gaussian)
      .export_values();

  py::enum_<ObservationOperatorType>(m, "ObservationOperatorType")
      .value("Identity", ObservationOperatorType::Identity)
      .value("Boundary", ObservationOperatorType::Boundary)
      .value("Sensors", ObservationOperatorType::Sensors)
      .value("SensorsGrid", ObservationOperatorType::SensorsGrid)
      .export_values();

  py::enum_<FEProductType>(m, "FEProductType")
      .value("L2", FEProductType::L2)
      .value("L2_0", FEProductType::L2_0)
      .value("H1_semi", FEProductType::H1_semi)
      .value("H1_0_semi", FEProductType::H1_0_semi)
      .value("H1", FEProductType::H1)
      .value("H1_0", FEProductType::H1_0)
      .value("Mass", FEProductType::Mass)
      .value("BoundaryMass", FEProductType::BoundaryMass)
      .export_values();

  py::enum_<ObservationSpaceProductType>(m, "ObservationSpaceProductType")
      .value("EUCLID", ObservationSpaceProductType::EUCLID)
      .value("STATE_L2", ObservationSpaceProductType::STATE_L2)
      .value("STATE_L2_0", ObservationSpaceProductType::STATE_L2_0)
      .value("STATE_H1_semi", ObservationSpaceProductType::STATE_H1_semi)
      .value("STATE_H1_0_semi", ObservationSpaceProductType::STATE_H1_0_semi)
      .value("STATE_H1", ObservationSpaceProductType::STATE_H1)
      .value("STATE_H1_0", ObservationSpaceProductType::STATE_H1_0)
      .export_values();

  py::enum_<BoundaryConditionType>(m, "BoundaryConditionType")
      .value("AllNeumann", BoundaryConditionType::AllNeumann)
      .value("DirichletOnYandZ", BoundaryConditionType::DirichletOnYandZ)
      .export_values();

  // -------------------------- Config --------------------------

  py::class_<MaterialModelBaseConfig>(m, "MaterialModelBaseConfig")
      .def(py::init<>())
      .def_readwrite("nt", &MaterialModelBaseConfig::nt)
      .def_readwrite("T_initial", &MaterialModelBaseConfig::T_initial)
      .def_readwrite("T_final", &MaterialModelBaseConfig::T_final)
      .def_readwrite("p1", &MaterialModelBaseConfig::p1)
      .def_readwrite("p2", &MaterialModelBaseConfig::p2)
      .def_readwrite("param_grid_resolution", &MaterialModelBaseConfig::param_grid_resolution)
      .def_readwrite("state_grid_resolution", &MaterialModelBaseConfig::state_grid_resolution)
      .def_readwrite("body_force_type", &MaterialModelBaseConfig::body_force_type)
      .def_readwrite("body_force_hyperparameter", &MaterialModelBaseConfig::body_force_hyperparameter)
      .def_readwrite("BC_type", &MaterialModelBaseConfig::BC_type)
      .def_readwrite("BC_hyperparameter", &MaterialModelBaseConfig::BC_hyperparameter);
}