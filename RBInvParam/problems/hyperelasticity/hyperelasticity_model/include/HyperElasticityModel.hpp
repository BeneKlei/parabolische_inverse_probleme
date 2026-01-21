#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <deal.II/base/exceptions.h>
#include <deal.II/base/array_view.h>

#include "MaterialModel.hpp"
#include "StoredEnergyFunction.hpp"
#include "StoredEnergyOperator.hpp"

struct HyperElasticityModelConfig : MaterialModelBaseConfig {
    StoredEnergyFunctionType se_type = StoredEnergyFunctionType::NeoHookean;
    StoredEnergyFunctionHyperparameter se_hyperparameter = {};
};

typedef double Number;
class HyperElasticityModel : public MaterialModel
{
public:
    using StorEneOp = StoredEnergyOperator<dim, Number>;

    explicit HyperElasticityModel(const HyperElasticityModelConfig& config);
    void setup_material_operator();

    std::unique_ptr<StorEneOp> assemble_A_q(
        const py::array_t<float, py::array::c_style | py::array::forcecast>& q_np
    );

    bool m_has_translation_operator = true;

private:
    // TODO Make Unique
    std::unique_ptr<StoredEnergyFunction<dim, Number>> m_stored_energy_function;
    bool m_q_time_dep = false;

    const HyperElasticityModelConfig m_hyperelasticity_config;

};
