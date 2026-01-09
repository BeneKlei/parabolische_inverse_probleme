#include "ElasticityModel.hpp"

ElasticityModel::ElasticityModel(const ElasticityModelConfig& config)
    : MaterialModel(static_cast<const MaterialModelBaseConfig&>(config))  // pass base part
    , m_elasticity_config(config)                                     // keep extended fields
{}

void ElasticityModel::setup_system_operator() 
{
    std::cout << "\t Setting up system matrizies." << std::endl;

    MaterialOperatorFactoryContext<dim, Number> ctx {
        m_elasticity_config.system_operator_type,
        m_fe,
        m_dof_handler,
        m_BC_constraints,
        m_system_matrix_sp,
        m_elasticity_config.system_operator_hyperparameter
    };

    std::vector<MatrixStack<Number>::MatV> _matrices;
    bool _affine;

    m_material_matrices_factory.assemble_system(
        ctx,
        _matrices,
        _affine
    );

    m_matrix_stack = std::make_shared<MatrixStack<Number>>(
        std::move(_matrices),
        m_system_matrix_sp,
        _affine
    );
    
    m_param_space_dim = m_matrix_stack->dim_Q();
    m_q.reinit(m_param_space_dim);
    m_state_space_dim = m_dof_handler.n_dofs();

    std::cout << "\t ---------------------- " << std::endl;
    std::cout << "\t #DoFs: " << m_state_space_dim  << std::endl;
    std::cout << "\t #Parameter: " << m_param_space_dim  << std::endl;
}

void ElasticityModel::assemble_A_q(py::array_t<float, py::array::c_style | py::array::forcecast> q_np) 
{   
    if (q_np.ndim() != 1)
        throw std::runtime_error("q must be a 1D numpy array");

    const auto buf = q_np.request();
    const std::size_t n = static_cast<std::size_t>(buf.size);
    const auto *ptr = static_cast<const float *>(buf.ptr);

    AssertDimension(n, m_matrix_stack->dim_Q());

    ArrayView<const float> q_view(ptr, n);

    MatrixStack<Number>::MatV matrix;
    m_matrix_stack->materialize(matrix, q_view);

    m_A_q = std::make_unique<MatrixOperator<Number>>(
        std::move(matrix),
        m_system_matrix_sp
    );
}

void ElasticityModel::assemble_partial_u_A_q_u(py::array_t<float, py::array::c_style | py::array::forcecast> q_np) 
{   
    if (q_np.ndim() != 1)
        throw std::runtime_error("q must be a 1D numpy array");

    const auto buf = q_np.request();
    const std::size_t n = static_cast<std::size_t>(buf.size);
    const auto *ptr = static_cast<const float *>(buf.ptr);

    AssertDimension(n, m_matrix_stack->dim_Q());

    ArrayView<const float> q_view(ptr, n);

    MatrixStack<Number>::MatV matrix;
    m_matrix_stack->materialize(matrix, q_view);

    m_A_q = std::make_unique<MatrixOperator<Number>>(
        std::move(matrix),
        m_system_matrix_sp
    );
}

void ElasticityModel::assemble_partial_u_A_q_u(py::array_t<float, py::array::c_style | py::array::forcecast> q_np) 
{   
    if (q_np.ndim() != 1)
        throw std::runtime_error("q must be a 1D numpy array");

    const auto buf = q_np.request();
    const std::size_t n = static_cast<std::size_t>(buf.size);
    const auto *ptr = static_cast<const float *>(buf.ptr);

    AssertDimension(n, m_matrix_stack->dim_Q());

    ArrayView<const float> q_view(ptr, n);

    MatrixStack<Number>::MatV matrix;
    m_matrix_stack->materialize(matrix, q_view);

    m_A_q = std::make_unique<MatrixOperator<Number>>(
        std::move(matrix),
        m_system_matrix_sp
    );
}

// void ElasticityModel::set_q(py::array_t<float, py::array::c_style | py::array::forcecast> q_np)
// {
//     if (q_np.ndim() != 1)
//         throw std::runtime_error("q must be a 1D numpy array");

//     auto buf = q_np.request();
//     const auto n = static_cast<std::size_t>(buf.size);
//     const float* ptr = static_cast<float*>(buf.ptr);

//     m_q.reinit(n);
//     std::copy(ptr, ptr + n, m_q.begin());   // one copy
// }
