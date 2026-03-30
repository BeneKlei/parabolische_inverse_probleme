// #pragma once

// #include <deal.II/base/function.h>
// #include <deal.II/base/point.h>

// #include <deal.II/lac/vector.h>

// using namespace dealii;

// typedef std::variant<int, double, std::string> BodyForceHyperparameterType;
// typedef std::map<std::string, BodyForceHyperparameterType>  BodyForceHyperparameter;

// enum class BodyForceType {
//     CenterExcite,
//     Dummy
// };

// class BodyForce : public Function<3>
// {
// public:
//   BodyForce() : Function<3>(3) {};
//   virtual ~BodyForce() = default;
//   virtual void vector_value(const Point<3> &p,
//                             Vector<double> &values) const override = 0;

//   void vector_value_list(const std::vector<Point<3>> &points,
//                          std::vector<Vector<double>> &value_list) const override;
// };

// class SharpPulseBodyForce : public BodyForce
// {
// public:
//   void vector_value(const Point<3> &p,
//                     Vector<double> &values) const override;
// };



// class GaussianPulseBodyForce : public BodyForce {
// public:
//     const Point<3> &center;
//     const float width;

//     // Constructor
//     GaussianBodyForce(const Point<3> &center_, float width_)
//         : center(center_), width(width_) {}

//     // Override vector_value
//     void vector_value(const Point<3> &p, Vector<double> &values) const override;
// };

