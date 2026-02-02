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
    using SEOp       = StoredEnergyOperator<dim, Number>;
    using SEJacOp    = StoredEnergyJacobianOperator<dim, Number>;
    using SEParamOp  = StoredEnergyParamDerivOperator<dim, Number>;
    using SpasMatOp = SparseMatrixOperator<Number>;

    explicit HyperElasticityModel(const HyperElasticityModelConfig& config);
    void setup_material_operator();

    // TODO Mkae q_np const
    std::unique_ptr<SEOp> assemble_A_q(
        const py::object q_np
    );

    // std::unique_ptr<SEJacOp> assemble_partial_u_A_q_u(
    //     const py::object q_np,
    //     const Vector<Number>& u
    // );

    std::unique_ptr<SEParamOp> assemble_partial_q_A_q_u(
        const py::object q_np,
        const Vector<Number>& u
    );

private:
    void _unpack_q_1d(
        const py::array_t<float, py::array::c_style | py::array::forcecast>& q_np,
        Vector<Number> &q
    ) const;

    std::unique_ptr<StoredEnergyFunction<dim, Number>> m_stored_energy_function;
    bool m_q_time_dep = false;

    const HyperElasticityModelConfig m_hyperelasticity_config;

};
