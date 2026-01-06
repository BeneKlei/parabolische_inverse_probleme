#include "ElasticityModel.hpp"


ElasticityModel::ElasticityModel(const MaterialModelConfig& config) : MaterialModel(config) {
    MaterialMatricesFactoryContext<3, Number> ctx {
        m_config.system_matrix_type,
        m_fe,
        m_dof_handler,
        m_BC_constraints,
        m_system_matrix_sp,
        m_config.system_matrix_hyperparameter
    };

    std::vector<MatrixStack::MatV> _matrices;
    bool _affine;

    m_material_matrices_factory.assemble_system(
        ctx,
        _matrices,
        _affine
    );

    m_matrix_stack = std::make_shared<MatrixStack<Number>>(
        _matrices,
        m_system_matrix_sp,
        _affine
    )    
}

void ElasticityModel::assemble_system_operator(Vector<Number> q) 
{
    m_op = std::make_unique<BilinearAqOp<Number>>(
        m_matrix_stack,
        q,
    )
}