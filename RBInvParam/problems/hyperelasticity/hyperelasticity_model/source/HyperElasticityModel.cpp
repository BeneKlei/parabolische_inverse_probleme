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

        m_stored_energy_function = std::make_unique<NeoHookeanStoredEnergy<dim, Number>>(
            mu,
            kappa,
            m_param_fe,
            m_param_dof_handler,
            m_dof_handler
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
    py::buffer_info buf;
    ArrayView<const float> q_view;
    _unpack_q_1d(q_np, buf, q_view);
    
    return std::make_unique<HyperElasticityModel::StorEneOp>(
       *m_stored_energy_function,
       m_fe,
       m_dof_handler
    );
}

void HyperElasticityModel::_unpack_q_1d(
  const py::array_t<float, py::array::c_style | py::array::forcecast>& q_np,
  py::buffer_info &buffer,
  ArrayView<const float> &q_view) const
{
  if (q_np.ndim() != 1)
    throw std::runtime_error("q must be a 1D numpy array");

  buffer = q_np.request();

  const std::size_t n = static_cast<std::size_t>(buffer.size);
  AssertDimension(n,m_param_dim);

  const auto *ptr = static_cast<const float *>(buffer.ptr);
  q_view = ArrayView<const float>(ptr, n);
}