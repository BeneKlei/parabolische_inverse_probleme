#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>

#include <variant>

using namespace dealii;

template <int dim, typename Number>
class StoredEnergyFunction
{
public:
    using PointType  = Point<dim, Number>;
    using TensorType = Tensor<2, dim, Number>;

    virtual ~StoredEnergyFunction() = default;

    virtual Number value(const PointType &p, const TensorType &F) const = 0;
    virtual TensorType gradient(const PointType &p, const TensorType &F) const = 0;
    virtual TensorType hessian(const PointType &p, const TensorType &F) const = 0;
    virtual TensorType contracted_hessian(const PointType &p, const TensorType &F,
                                        const TensorType &H) const = 0;

    using TensorTypeMinus1 = Tensor<1, dim, Number>;
    virtual TensorType contracted_hessian(const PointType &p, const TensorType &F,
                                        const TensorTypeMinus1 &H,
                                        std::size_t comp) const = 0;
};

// -------------------------------------------------------------------------------

enum class StoredEnergyFunctionType {
    NeoHookean,
    Hookean
};

typedef std::variant<int, double, std::string> StoredEnergyFunctionHyperparameterType;
typedef std::map<std::string, StoredEnergyFunctionHyperparameterType>  StoredEnergyFunctionHyperparameter;

// ----------------------------------- HookeanStoredEnergy -----------------------------------

template <int dim>
class HookeanStoredEnergy : public StoredEnergyFunction<dim, double>
{
public:
    using Base             = StoredEnergyFunction<dim, double>;
    using PointType        = typename Base::PointType;        
    using TensorType       = typename Base::TensorType;       
    using TensorTypeMinus1 = typename Base::TensorTypeMinus1; 

    HookeanStoredEnergy(
        double lambda, 
        double mu
    );

    double value(const PointType &p, const TensorType &F) const override;

    TensorType gradient(const PointType &p, const TensorType &F) const override;
    TensorType hessian(const PointType &p, const TensorType &F) const override;

    TensorType contracted_hessian(const PointType &p,
                                  const TensorType &F,
                                  const TensorType &H) const override;

    TensorType contracted_hessian(const PointType &p,
                                  const TensorType &F,
                                  const TensorTypeMinus1 &H,
                                  std::size_t comp) const override;

private:
    double m_lambda;
    double m_mu;
};

// ---------------------------------- NeoHookeanStoredEnergy ----------------------------------


template <int dim>
class NeoHookeanStoredEnergy : public StoredEnergyFunction<dim, double>
{
public:
    using Base             = StoredEnergyFunction<dim, double>;
    using PointType        = typename Base::PointType;        
    using TensorType       = typename Base::TensorType;       
    using TensorTypeMinus1 = typename Base::TensorTypeMinus1; 

    NeoHookeanStoredEnergy(
        double mu, 
        double kappa
    );

    double value(const PointType &p, const TensorType &F) const override;

    TensorType gradient(const PointType &p, const TensorType &F) const override;
    TensorType hessian(const PointType &p, const TensorType &F) const override;

    TensorType contracted_hessian(const PointType &p,
                                  const TensorType &F,
                                  const TensorType &H) const override;

    TensorType contracted_hessian(const PointType &p,
                                  const TensorType &F,
                                  const TensorTypeMinus1 &H,
                                  std::size_t comp) const override;

private:
    double m_mu;
    double m_kappa;
    double m_beta;
    double m_c1;

    const double m_tol_gradient = 1e-12;
    const double m_tol_hessian = 1e-12;
};
