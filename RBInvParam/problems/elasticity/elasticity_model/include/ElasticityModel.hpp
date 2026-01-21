#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <deal.II/base/exceptions.h>
#include <deal.II/base/array_view.h>

#include "MaterialModel.hpp"
#include "MatrixOperator.hpp"
#include "MaterialOperatorFactory.hpp"


struct ElasticityModelConfig : MaterialModelBaseConfig {
    MaterialOperatorType material_operator_type = MaterialOperatorType::CosseratDelamination;
    MaterialOperatorHyperparameter material_operator_hyperparameter = {};
};

class ElasticityModel : public MaterialModel
{
public:
    using SpasMatOp = SparseMatrixOperator<Number>;
    using FullMatOp = FullMatrixOperator<Number>;

    explicit ElasticityModel(const ElasticityModelConfig& config);
    
    void setup_material_operator();

    std::unique_ptr<SpasMatOp> assemble_A_q(
        const py::array_t<float, py::array::c_style | py::array::forcecast>& q_np,
        bool linear_part_only = false
    );
    std::unique_ptr<FullMatOp> assemble_partial_q_A_q_u(
        const Vector<Number>& u
    );
    std::unique_ptr<SpasMatOp> assemble_partial_u_A_q_u(
        const py::array_t<float, py::array::c_style | py::array::forcecast>& q_np
    );

    std::unique_ptr<SpasMatOp> get_translation_operator();
    bool m_has_translation_operator = true;

private:
    const ElasticityModelConfig& m_elasticity_config;
    bool m_q_time_dep = false;

    std::unique_ptr<MatrixStack<Number>> m_matrix_stack;
    MaterialOperatorFactory<dim, Number> m_material_matrices_factory = MaterialOperatorFactory<3, Number>();
};

