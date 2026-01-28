#include "HyperElasticityModel.hpp"

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

};

std::unique_ptr<typename HyperElasticityModel::StorEneOp> 
HyperElasticityModel::assemble_A_q(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& q_np
) 
{
    Vector<Number> q;
    _unpack_q_1d(q_np, q);
    
    return std::make_unique<HyperElasticityModel::StorEneOp>(
        q,
        this->state_space_context(),
        this->param_space_context(),
        *m_stored_energy_function
    );
}

void HyperElasticityModel::_unpack_q_1d(
  const py::array_t<float, py::array::c_style | py::array::forcecast>& q_np,
  Vector<Number> &q) const
{
  if (q_np.ndim() != 1)
    throw std::runtime_error("q must be a 1D numpy array");

  py::buffer_info buffer = q_np.request();

  const std::size_t n = static_cast<std::size_t>(buffer.size);
  AssertDimension(n,m_param_dim);
  q.reinit(n);

  const auto *ptr = static_cast<const float *>(buffer.ptr);
  std::transform(ptr, ptr + n, q.begin(),
                 [](float v) { return static_cast<double>(v); });
}