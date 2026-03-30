#pragma once

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/dofs/dof_handler.h>

using namespace dealii;

typedef std::variant<int, double, std::string, std::vector<double>> BodyForceHyperparameterType;
typedef std::map<std::string, BodyForceHyperparameterType>  BodyForceHyperparameter;
// ---------------------------------------------------------------------------------------------------------------------

// template <class T>
// inline void check_required_keys(
//     const BodyForceHyperparameter& params,
//     const std::initializer_list<std::string>& required_keys) 
// {
//     for (const auto& key : required_keys) {
//         auto it = params.find(key);
//         if (it == params.end()) {
//             throw std::runtime_error("Missing key: " + key);
//         }
//         if (!std::holds_alternative<T>(it->second)) {
//             throw std::runtime_error(
//                 "Key '" + key + "' must be of type " + std::string(type_name<T>())
//             );
//         }
//     }
// }

// ---------------------------------------------------------------------------------------------------------------------

enum class BodyForceType {
    SharpPulse,
    WavePulse,
    GaussianPulse
};

template <int dim, typename Number>
struct BodyForceFactoryContext {
  const BodyForceType                &body_force_type;
  const FiniteElement<dim>           &fe;
  const DoFHandler<dim>              &dof_handler;
  const BodyForceHyperparameter      &hyperparameter;
};

// ---------------------------------------------------------------------------------------------------------------------

class BodyForce : public Function<3>
{
public:
  BodyForce() : Function<3>(3) {};
  virtual ~BodyForce() = default;
  virtual void vector_value(const Point<3> &p,
                            Vector<double> &values) const override = 0;

  void vector_value_list(const std::vector<Point<3>> &points,
                         std::vector<Vector<double>> &value_list) const override;
};

class SharpPulseBodyForce : public BodyForce
{
public:
    const Point<3> origin;
    const double factor;
    const double end_time;
    // Constructor
    SharpPulseBodyForce(const Point<3> &origin_, double &end_time_, double &factor_)
        : origin(origin_)
        , end_time(end_time_)
        , factor(factor_)
        {}

    void vector_value(const Point<3> &p,
                      Vector<double> &values) const override;
};

class WavePulseBodyForce : public BodyForce
{
public:
    const Point<3> origin;
    const double factor;
    const double end_time;
    const double time_scaling_factor;
    // Constructor
    WavePulseBodyForce(
        const Point<3> &origin_, 
        double &end_time_, 
        double &factor_,
        double &time_scaling_factor_
    )
        : origin(origin_)
        , end_time(end_time_)
        , factor(factor_)
        , time_scaling_factor(time_scaling_factor_)
        {}

    void vector_value(const Point<3> &p,
                      Vector<double> &values) const override;
};

class GaussianPulseBodyForce : public BodyForce {
public:
    const Point<3> origin;
    const double width;
    const double end_time; 

    // Constructor
    GaussianPulseBodyForce(const Point<3> &origin_, double &width_, double &end_time_)
        : origin(origin_)
        , width(width_) 
        , end_time(end_time_)
        {}

    // Override vector_value
    void vector_value(const Point<3> &p, Vector<double> &values) const override;
};


// ---------------------------------------------------------------------------------------------------------------------

template <int dim, typename Number>
class BodyForceFactory
{   
public:
    std::unique_ptr<BodyForce> assemble_body_force(
        const BodyForceFactoryContext<dim, Number>& ctx
    ) const;

    std::unique_ptr<BodyForce> assemble_sharp_pulse_body_force(
        const BodyForceFactoryContext<dim, Number>& ctx
    ) const;

    std::unique_ptr<BodyForce> assemble_wave_pulse_body_force(
        const BodyForceFactoryContext<dim, Number>& ctx
    ) const;

    std::unique_ptr<BodyForce> assemble_gaussian_pulse_body_force(
        const BodyForceFactoryContext<dim, Number>& ctx
    ) const;

};
