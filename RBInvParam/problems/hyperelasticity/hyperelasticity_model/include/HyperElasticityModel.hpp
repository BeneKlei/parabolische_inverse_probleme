#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <deal.II/base/exceptions.h>
#include <deal.II/base/array_view.h>

#include "MaterialModel.hpp"
#include "StoredEnergyFunction.hpp"
#include "StoredEnergyOperator.hpp"
#include "MatrixOperator.hpp"

struct HyperElasticityModelConfig : MaterialModelBaseConfig {
    StoredEnergyFunctionType se_type = StoredEnergyFunctionType::NeoHookean;
    StoredEnergyFunctionHyperparameter se_hyperparameter = {};
};

typedef double Number;
class HyperElasticityModel : public MaterialModel
{
public:
    using BaseOp         = BaseOperator<Number>;
    using FullMatOp      = FullMatrixOperator<Number>;

    using LinSEOp        = LinearStoredEnergyOperator<dim, Number>;
    using SEOp           = StoredEnergyOperator<dim, Number>;
    using SEParamDerivOp = StoredEnergyParamDerivOperator<dim, Number>;

    explicit HyperElasticityModel(const HyperElasticityModelConfig& config);
    void setup_material_operator();

    std::unique_ptr<BaseOp> assemble_A_q(
        const py::object q_np,
        bool param_linear_part_only = false
    );

    std::unique_ptr<BaseOp> _assemble_A_q(
        const Vector<Number>& q,
        bool param_linear_part_only = false
    );

    std::unique_ptr<BaseOp> assemble_partial_q_A_q_u(
        const py::object q_np,
        const Vector<Number>& u
    );

private:
    void _unpack_q_1d(
        const py::array_t<float, py::array::c_style | py::array::forcecast>& q_np,
        Vector<Number> &q
    ) const;

    std::unique_ptr<StoredEnergyFunction<dim, Number>> m_stored_energy_function;
    const HyperElasticityModelConfig m_hyperelasticity_config;

};
