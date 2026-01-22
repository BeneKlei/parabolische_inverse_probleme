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

template <int dim, typename Number>
class ParametricStoredEnergyFunction : public StoredEnergyFunction<dim, Number>
{
public: 
    void set_param(ArrayView<const float> param_view) {
        m_param.assign(param_view.begin(), param_view.end());
    }

    Vector<Number> m_param;
};


// -------------------------------------------------------------------------------

enum class StoredEnergyFunctionType {
    NeoHookean
};

typedef std::variant<int, double, std::string> StoredEnergyFunctionHyperparameterType;
typedef std::map<std::string, StoredEnergyFunctionHyperparameterType>  StoredEnergyFunctionHyperparameter;


// TODO Fix number to double
template <int dim, typename Number>
class NeoHookeanStoredEnergy : public ParametricStoredEnergyFunction<dim, Number>
{
public:
    using Base             = StoredEnergyFunction<dim, Number>;
    using PointType        = typename Base::PointType;        
    using TensorType       = typename Base::TensorType;       
    using TensorTypeMinus1 = typename Base::TensorTypeMinus1; 

    NeoHookeanStoredEnergy(
        double mu, 
        double kappa,
        const FiniteElement<dim>& param_fe,
        const DoFHandler<dim>& param_dof_handler,
        const DoFHandler<dim>& state_dof_handler
    );

    Number value(const PointType &p, const TensorType &F) const override;

    TensorType gradient(const PointType &p, const TensorType &F) const override;
    TensorType hessian(const PointType &p, const TensorType &F) const override;

    TensorType contracted_hessian(const PointType &p,
                                  const TensorType &F,
                                  const TensorType &H) const override;

    TensorType contracted_hessian(const PointType &p,
                                  const TensorType &F,
                                  const TensorTypeMinus1 &H,
                                  std::size_t comp) const override;

    //void setup_field_function();

private:
    double compute_param_factor(const PointType& p) const;
    //bool in_first_layer(const PointType& p) const;


    double m_mu;
    double m_kappa;
    double m_beta;
    double m_c1;
    
    const FiniteElement<dim>& m_param_fe;
    const DoFHandler<dim>& m_param_dof_handler;
    const DoFHandler<dim>& m_state_dof_handler;
    const MappingQ1<dim> m_mapping;
    //Functions::FEFieldFunction<dim> m_param_factor_field;
};


// #include <deal.II/base/point.h>
// #include <deal.II/base/tensor.h>

// #include <deal.II/fe/mapping_q1.h>
// #include <deal.II/grid/grid_tools.h>

// #include <deal.II/numerics/fe_field_function.h>
// #include <deal.II/numerics/vector_tools.h>

// using namespace dealii;


// FE_Q<3>       m_fe_alpha;       // FE_Q(1)
// DoFHandler<3> m_dh_alpha;
// Vector<double> m_alpha;

// // alpha (scalar)
// FE_Q<3>       m_fe_alpha;       // FE_Q(1)
// DoFHandler<3> m_dh_alpha;
// Vector<double> m_alpha;



// // Your constitutive law: returns Chat(x, Y). Replace with your real one.
// // Here Y = grad u (Tensor<2,3>) as an example.
// double Chat(const Point<3> &x, const Tensor<2,3> &Y)
// {
//     (void)x;
//     (void)Y;
//     return 1.0; // placeholder
// }

// // point-based "first layer": cell containing x touches boundary_id == 1
// bool in_first_layer(const DoFHandler<3> &dof_handler_any,
//                     const Point<3> &x)
// {
//     MappingQ1<3> mapping;

//     const auto cell_and_ref =
//         GridTools::find_active_cell_around_point(mapping, dof_handler_any, x);

//     const auto &cell = cell_and_ref.first;

//     for (unsigned int f = 0; f < GeometryInfo<3>::faces_per_cell; ++f)
//         if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 1)
//             return true;

//     return false;
// // }

// // Compute C_alpha(x, Y(x)) at arbitrary x
// double C_alpha_at_point(const DoFHandler<3> &dof_handler_u,
//                         const Vector<double> &u,                 // displacement dofs
//                         const DoFHandler<3> &dof_handler_alpha,
//                         const Vector<double> &alpha,             // alpha dofs
//                         const Point<3> &x)
// {
//     MappingQ1<3> mapping;

//     // 1) Y(x) from displacement field.
//     // For u in H^1(Omega;R^3), Y(x) is typically grad u(x) (3x3 tensor).
//     const Tensor<2,3> Y =
//         VectorTools::point_gradient(mapping, dof_handler_u, u, x);

//     // 2) Compute Chat(x, Y(x))
//     const double Chat_x = Chat(x, Y);

//     // 3) If not in first layer, return undamaged constitutive response
//     if (!in_first_layer(dof_handler_u, x))
//         return Chat_x;

//     // 4) alpha_h(x) = sum_K alpha_K v_K(x)
//     Functions::FEFieldFunction<3> alpha_field(mapping, dof_handler_alpha, alpha);
//     const double alpha_x = alpha_field.value(x);

//     // 5) Damaged response
//     return alpha_x * Chat_x;
// }