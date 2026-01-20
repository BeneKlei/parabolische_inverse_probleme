#include "HyperElasticityModel.hpp"

template <int dim, typename Number>
HyperElasticityModel<dim, Number>::HyperElasticityModel(const HyperElasticityModelConfig& config)
    : MaterialModel(static_cast<const MaterialModelBaseConfig&>(config))
    , m_hyperelasticity_config(config)                                  
{    

    switch (m_hyperelasticity_config.se_type)
    {
    case StoredEnergyFunctionType::NeoHookean:
        std::cout << "\t Using NeoHookean stored energy function" << std::endl;        

        double mu    = std::get<double>(m_hyperelasticity_config.se_hyperparameter.at("mu"));
        double kappa = std::get<double>(m_hyperelasticity_config.se_hyperparameter.at("kappa"));

        m_stored_energy_function = NeoHookeanStoredEnergy<dim, Number>(
            mu = mu,
            kappa = kappa
        );
    default:
        throw std::runtime_error("Unknown StoredEnergyFunctionType.");
    }
    
    m_param_space_dim = 0;
    m_state_space_dim = m_dof_handler.n_dofs();

    std::cout << "\t ---------------------- " << std::endl;
    std::cout << "\t #DoFs: " << m_state_space_dim  << std::endl;
    std::cout << "\t #Parameter: " << m_param_space_dim  << std::endl;
}

template <int dim, typename Number>
std::unique_ptr<typename HyperElasticityModel<dim, Number>::StorEneOp> 
HyperElasticityModel<dim, Number>::assemble_A_q(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& q_np
) 
{
    StoredEnergyOperatorContext<dim> ctx {
        m_fe,
        m_dof_handler
    };

    return std::make_unique<HyperElasticityModel<dim, Number>::StorEneOp>(
       m_stored_energy_function,
       ctx
    );
}