#include <vector>

#include <deal.II/base/tensor.h>

class StoredEnergy 
{
public:
    /* TODOs:
        - How can this be vectorized? 
        - Why is H <1,3>?
    */

    virtual void stored_energy(dealii::Tensor<2,3> &F, 
                               double &ret_vals) = 0;

    // virtual void DY_C(dealii::Tensor<2,3> F, 
    //                   std::vector<dealii::Tensor<2,3>> &ret_vals) = 0;

    // virtual void DY_DY_C_H(dealii::Tensor<2,3> F, 
    //                        dealii::Tensor<1,3> H, 
    //                        const unsigned int comp_i,
    //                        std::vector<dealii::Tensor<2,3>> &ret_vals) = 0; // DY_DY_C(Y):H

    void slice_H(dealii::Tensor<1,3> H, 
                 const unsigned int comp_i);

protected:
    double J;
    double I1;
    dealii::Tensor<2,3> FT;
    dealii::Tensor<2,3> FMT;
    dealii::Tensor<2,3> FHF;
    dealii::Tensor<4,3> YxY;
    dealii::Tensor<2,3> YxY_H;

    dealii::Tensor<2,3> sliced_H;
};

class NeoHookianStoredEnergy : public StoredEnergy {
public:
    NeoHookianStoredEnergy(double mu_, 
                           double kappa_);
    ~NeoHookianStoredEnergy() {};

    void stored_energy(dealii::Tensor<2,3> &F, double &ret_vals) override;

private:
    double mu;
    double kappa;
    double beta;
    double c_1;
};