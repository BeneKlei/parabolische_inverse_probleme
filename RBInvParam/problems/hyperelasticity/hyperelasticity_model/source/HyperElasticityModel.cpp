#include "HyperElasticityModel.hpp"
#include "utils.hpp"


HyperElasticityModel::HyperElasticityModel(const HyperElasticityModelConfig& config)
    : MaterialModel(static_cast<const MaterialModelBaseConfig&>(config))
    , m_hyperelasticity_config(config)                                  
{}

void HyperElasticityModel::setup_material_operator() {

    switch (m_hyperelasticity_config.se_type)
    {
    case StoredEnergyFunctionType::NeoHookean: 
    {
        std::cout << "\t Using NeoHookean stored energy function" << std::endl;

        double mu    = std::get<double>(m_hyperelasticity_config.se_hyperparameter.at("mu"));
        double kappa = std::get<double>(m_hyperelasticity_config.se_hyperparameter.at("kappa"));

        m_stored_energy_function = std::make_unique<NeoHookeanStoredEnergy<dim>>(
            mu,
            kappa
        );
        break;
    }
    case StoredEnergyFunctionType::Hookean: 
    {
        std::cout << "\t Using Hookean stored energy function" << std::endl;

        double mu    = std::get<double>(m_hyperelasticity_config.se_hyperparameter.at("mu"));
        double lambda = std::get<double>(m_hyperelasticity_config.se_hyperparameter.at("lambda"));

        m_stored_energy_function = std::make_unique<HookeanStoredEnergy<dim>>(
            mu,
            lambda
        );
        break;
    }
    default:
        throw std::runtime_error("Unknown StoredEnergyFunctionType.");
    }

    m_A_q_linear = m_stored_energy_function->m_linear;
    m_A_affine = true;

    const auto &ts = m_state_space_context.triangulation();
    const auto &tp = m_param_space_context.triangulation();

    if (!utils::same_tria_geometry_and_connectivity<dim>(ts, tp, /*tol=*/1e-14))
        throw std::runtime_error("State and parameter triangulation must be equivalent (same mesh).");

    if (!utils::same_quadrature<dim>(m_state_space_context.quadrature(),
                                     m_param_space_context.quadrature(),
                                     /*tol=*/1e-14))
        throw std::runtime_error("State and parameter quadrature must be equivalent.");

    if (!utils::same_mapping_configuration<dim>(m_state_space_context.mapping(),
                                                m_param_space_context.mapping()))
        throw std::runtime_error("State and parameter mapping must be equivalent.");
};

std::unique_ptr<typename HyperElasticityModel::BaseOp> 
HyperElasticityModel::assemble_A_q(
    py::object q_np,
    bool param_linear_part_only
) 
{
    Vector<Number> q;
    if (!q_np.is_none())
      _unpack_q_1d(q_np.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(), q);
    
    return _assemble_A_q(q, param_linear_part_only);
}

std::unique_ptr<typename HyperElasticityModel::BaseOp> 
HyperElasticityModel::_assemble_A_q(
    const Vector<Number>& q,
    bool param_linear_part_only
) 
{
    Vector<Number> full_q;
    
    if (m_A_q_linear) 
        return std::make_unique<HyperElasticityModel::LinSEOp>(
            q,
            full_q,
            this->m_state_space_context,
            this->m_param_space_context,
            *m_stored_energy_function,
            param_linear_part_only
        );
    else 
        return std::make_unique<HyperElasticityModel::SEOp>(
            q,
            full_q,
            this->m_state_space_context,
            this->m_param_space_context,
            *m_stored_energy_function,
            param_linear_part_only
        );
}

std::unique_ptr<typename HyperElasticityModel::BaseOp> 
HyperElasticityModel::assemble_partial_q_A_q_u(
    py::object q_np,
    const Vector<Number>& u
)
{
    Vector<Number> q;
    Vector<Number> full_q;
    if (!q_np.is_none())
      _unpack_q_1d(q_np.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(), q);
    
    return std::make_unique<HyperElasticityModel::SEParamDerivOp>(
        u,
        q,
        full_q,
        this->m_state_space_context,
        this->m_param_space_context,
        *m_stored_energy_function
    );
}

// std::unique_ptr<typename HyperElasticityModel::BaseOp> 
// HyperElasticityModel::assemble_partial_q_A_q_u(
//     py::object q_np,
//     const Vector<Number>& u
// )
// {
//     Vector<Number> e(m_param_dim);
//     Vector<Number> A_q_u(m_state_dim);

//     FullMatrix<Number> matrix;
//     matrix.reinit(m_state_dim, m_param_dim);

//     std::unique_ptr<HyperElasticityModel::BaseOp> A_q;
    
//     for (unsigned int i = 0; i < m_param_dim; ++i)
//     {
//         e = Number(0.0);
//         e(i) = Number(1.0);
//         A_q = _assemble_A_q(e, true);
//         A_q->apply(A_q_u, u);

//         for (unsigned int j = 0; j < m_state_dim; ++j)
//             matrix(j, i) = A_q_u[j];
//     }

//     return std::make_unique<HyperElasticityModel::FullMatOp>(
//         std::move(matrix)
//     );
// }   



void HyperElasticityModel::_unpack_q_1d(
  const py::array_t<float, py::array::c_style | py::array::forcecast>& q_np,
  Vector<Number> &q) const
{
  if (q_np.ndim() != 1)
    throw std::runtime_error("q must be a 1D numpy array");

  py::buffer_info buffer = q_np.request();

  const std::size_t n = static_cast<std::size_t>(buffer.size);
  if (n == 0)
    return;

  AssertDimension(n, m_param_dim);
  q.reinit(n);

  const auto *ptr = static_cast<const float *>(buffer.ptr);
  std::transform(ptr, ptr + n, q.begin(),
                 [](float v) { return static_cast<double>(v); });
}
