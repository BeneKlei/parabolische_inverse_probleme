#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/lac/vector.h>

using namespace dealii;

enum class BodyForceType {
    CenterExcite,
    Dummy
};

// // Convert config string -> enum
// inline BodyForceType parse_body_force(std::string_view name) {
//     if (name == "center_excite") return BodyForceType::CenterExcite;
//     throw std::runtime_error(std::string("Unknown body force: ") + std::string(name));
// }

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

class CenterExciteBodyForce : public BodyForce
{
public:
  void vector_value(const Point<3> &p,
                    Vector<double> &values) const override;
};

