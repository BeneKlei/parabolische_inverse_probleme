#include "ElasticityModel.hpp"

ElasticityModel::ElasticityModel(const ElasticityModelConfig& config)
    : MaterialModel(static_cast<const MaterialModelBaseConfig&>(config))  // pass base part
    , m_elasticity_config(config)                                     // keep extended fields
{
    std::size_t reserve_size = 1;
    std::size_t _nt = (m_elasticity_config.nt + 1);

    if (m_q_time_dep) {
        reserve_size = _nt;
    }

    m_A_q.resize(reserve_size);
    m_partial_q_A_q_u.resize(reserve_size);
    m_partial_u_A_q_u.resize(_nt);

}

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
        _affine
    );
    
    m_param_space_dim = m_matrix_stack->dim_Q();
    m_q.reinit(m_param_space_dim);
    m_state_space_dim = m_dof_handler.n_dofs();

    std::cout << "\t ---------------------- " << std::endl;
    std::cout << "\t #DoFs: " << m_state_space_dim  << std::endl;
    std::cout << "\t #Parameter: " << m_param_space_dim  << std::endl;
}

void ElasticityModel::assemble_A_q(
    py::array_t<float, py::array::c_style | py::array::forcecast> q_np,
    std::size_t time_step
) 
{   
    
    AssertIndexRange(time_step, m_A_q.size());

    py::buffer_info buf;
    ArrayView<const float> q_view;
    _unpack_q_1d(std::move(q_np), buf, q_view);

    MatrixStack<Number>::MatV matrix;
    m_matrix_stack->materialize(matrix, q_view);

    m_A_q[time_step] = std::make_unique<ElasticityModel::SpasMatOp>(
        std::move(matrix)
    );
}

void ElasticityModel::assemble_partial_q_A_q_u(
    const Vector<Number>& u,
    std::size_t time_step
) 
{   
    AssertIndexRange(time_step, m_partial_q_A_q_u.size());
    AssertDimension(u.size(), m_matrix_stack->dim_V());

    std::vector<Vector<Number>> A_us;
    m_matrix_stack->apply_to_each_matrix(u, A_us, false);
    FullMatrix<Number> matrix;

    for (unsigned int j = 0; j < m_param_space_dim; ++j)
        for (unsigned int i = 0; i < m_state_space_dim; ++i)
            matrix(i, j) = A_us[j][i];
    
    m_partial_q_A_q_u[time_step] = std::make_unique<ElasticityModel::FullMatOp>(
        std::move(matrix)
    );
}

void ElasticityModel::assemble_partial_u_A_q_u(
    py::array_t<float, py::array::c_style | py::array::forcecast> q_np,
    std::size_t time_step
) 
{ 
    AssertIndexRange(time_step, m_partial_u_A_q_u.size());
    

    py::buffer_info buf;
    ArrayView<const float> q_view;
    _unpack_q_1d(std::move(q_np), buf, q_view);

    MatrixStack<Number>::MatV matrix;
    m_matrix_stack->materialize(matrix, q_view);

    m_partial_u_A_q_u[time_step] = std::make_unique<ElasticityModel::SpasMatOp>(
        std::move(matrix)
    );
}

void ElasticityModel::_unpack_q_1d(
  py::array_t<float, py::array::c_style | py::array::forcecast> q_np,
  py::buffer_info &buffer,
  ArrayView<const float> &q_view) const
{
  if (q_np.ndim() != 1)
    throw std::runtime_error("q must be a 1D numpy array");

  buffer = q_np.request();

  const std::size_t n = static_cast<std::size_t>(buffer.size);
  AssertDimension(n, m_matrix_stack->dim_Q());

  const auto *ptr = static_cast<const float *>(buffer.ptr);
  q_view = ArrayView<const float>(ptr, n);
}