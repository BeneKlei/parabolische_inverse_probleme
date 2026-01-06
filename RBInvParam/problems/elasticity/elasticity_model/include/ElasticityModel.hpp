#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "MaterialModel.hpp"
#include "BilinearOperator.hpp"
#include "MaterialOperatorFactory.hpp"


struct ElasticityModelConfig : MaterialModelBaseConfig {
    MaterialOperatorType system_operator_type = MaterialOperatorType::CosseratDelamination;
    SystemOperatorHyperparameter system_operator_hyperparameter = {};
};

class ElasticityModel : public MaterialModel
{
public:
    using Op = BilinearAqOp<Number>;
    explicit ElasticityModel(const ElasticityModelConfig& config);
    
    void setup_system_operator();
    void assemble_system_operator();
    void set_q(py::array_t<float, py::array::c_style | py::array::forcecast> q_np);

    std::unique_ptr<Op> m_op;

private:
    const ElasticityModelConfig& m_elasticity_config;

    std::shared_ptr<const MatrixStack<Number>> m_matrix_stack;
    Vector<Number> m_q;

    MaterialOperatorFactory<dim, Number> m_material_matrices_factory = MaterialOperatorFactory<3, Number>();
};

