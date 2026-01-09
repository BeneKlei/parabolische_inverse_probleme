#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <deal.II/base/array_view.h>

#include "MaterialModel.hpp"
#include "MatrixOperator.hpp"
#include "MaterialOperatorFactory.hpp"


struct ElasticityModelConfig : MaterialModelBaseConfig {
    MaterialOperatorType system_operator_type = MaterialOperatorType::CosseratDelamination;
    SystemOperatorHyperparameter system_operator_hyperparameter = {};
};

class ElasticityModel : public MaterialModel
{
public:
    using Op = MatrixOperator<Number>;
    explicit ElasticityModel(const ElasticityModelConfig& config);
    
    void setup_system_operator();
    void assemble_A_q(py::array_t<float, py::array::c_style | py::array::forcecast> q_np);
    void assemble_partial_q_A_q_u(py::array_t<float, py::array::c_style | py::array::forcecast> q_np);
    void assemble_partial_u_A_q_u(
        py::array_t<float, py::array::c_style | py::array::forcecast> q_np,
        
    );

    

    //void assemble_system_operators(py::array_t<float, py::array::c_style | py::array::forcecast> q_np);

    std::unique_ptr<Op> m_A_q;
    std::unique_ptr<Op> m_partial_q_A_q_u;
    std::unique_ptr<Op> m_partial_u_A_q_u;

private:
    const ElasticityModelConfig& m_elasticity_config;

    std::shared_ptr<const MatrixStack<Number>> m_matrix_stack;
    MaterialOperatorFactory<dim, Number> m_material_matrices_factory = MaterialOperatorFactory<3, Number>();
};

