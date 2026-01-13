#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <deal.II/base/exceptions.h>
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
    using SpasMatOp = SparseMatrixOperator<Number>;
    using FullMatOp = FullMatrixOperator<Number>;

    explicit ElasticityModel(const ElasticityModelConfig& config);
    
    void setup_system_operator();
    void assemble_A_q(
        py::array_t<float, py::array::c_style | py::array::forcecast> q_np,
        std::size_t time_step = 0
    );
    void assemble_partial_q_A_q_u(
        const Vector<Number>& u,
        std::size_t time_step = 0
    );
    void assemble_partial_u_A_q_u(
        py::array_t<float, py::array::c_style | py::array::forcecast> q_np,
        std::size_t time_step = 0
    );

    const SpasMatOp* get_A_q(std::size_t time_step = 0) const {
        AssertIndexRange(time_step, m_A_q.size());
        return m_A_q[time_step].get();
    }
    const FullMatOp* get_partial_q_A_q_u(std::size_t time_step = 0) const {
        AssertIndexRange(time_step, m_partial_q_A_q_u.size());
        return m_partial_q_A_q_u[time_step].get();
    }
    const SpasMatOp* get_partial_u_A_q_u(std::size_t time_step = 0) const {
        AssertIndexRange(time_step, m_partial_u_A_q_u.size());
        return m_partial_u_A_q_u[time_step].get();
    }

private:
    void _unpack_q_1d(
        py::array_t<float, py::array::c_style | py::array::forcecast> q_np,
        py::buffer_info &buffer,
        ArrayView<const float> &q_view
    ) const;

    std::vector<std::unique_ptr<SpasMatOp>> m_A_q;
    std::vector<std::unique_ptr<FullMatOp>> m_partial_q_A_q_u;
    std::vector<std::unique_ptr<SpasMatOp>> m_partial_u_A_q_u;


    const ElasticityModelConfig& m_elasticity_config;
    bool m_q_time_dep = false;

    std::shared_ptr<const MatrixStack<Number>> m_matrix_stack;
    MaterialOperatorFactory<dim, Number> m_material_matrices_factory = MaterialOperatorFactory<3, Number>();
};

