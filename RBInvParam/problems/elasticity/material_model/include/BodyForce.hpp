#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/lac/vector.h>

using namespace dealii;

class BodyForce : public Function<3> 
{
public:
    BodyForce();
    void vector_value(const Point<3> &p, Vector<double> &values) const override;
    void vector_value_list(const std::vector<Point<3>> &points, std::vector<Vector<double>> &value_list) const override;
};




