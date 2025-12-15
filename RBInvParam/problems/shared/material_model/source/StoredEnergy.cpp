#include <stdexcept>

#include "StoredEnergy.hpp"

// --------------------------------------------------------------------

void StoredEnergy::slice_H(dealii::Tensor<1,3> H, 
                           const unsigned int comp_i) 
{
    for(unsigned int k = 0; k < 3; k++) {
        if(k == comp_i) {
            sliced_H[k] = H;
        } else {
            sliced_H[k] = 0;
        }
    }
};

// --------------------------------------------------------------------

NeoHookianStoredEnergy::NeoHookianStoredEnergy(double mu_, 
                                               double kappa_) : mu(mu_), kappa(kappa_) 
{
    beta = (3.0*kappa-2.0*mu)/(6.0*mu);
    c_1 = mu/2.0;
};

void NeoHookianStoredEnergy::stored_energy(dealii::Tensor<2,3> &F, 
                                           double &ret_vals) 
{
    J = determinant(F);
    I1 = trace(transpose(F)*F);    
    ret_vals = c_1 * (I1 - 3.0) + (c_1)/(beta) * (std::pow(J,-2.0*beta) - 1.0);
};

// void NeoHookianStoredEnergy::DY_C(dealii::Tensor<2,3> F, 
//                                   std::vector<dealii::Tensor<2,3>> &ret_vals) 
// {
//     throw std::logic_error("Not implemented yet");
// };

// void NeoHookianStoredEnergy::DY_DY_C_H(dealii::Tensor<2,3> F, 
//                                        dealii::Tensor<1,3> H, 
//                                        const unsigned int comp_i,
//                                        std::vector<dealii::Tensor<2,3>> &ret_vals) 
// {
//     throw std::logic_error("Not implemented yet");
// };
