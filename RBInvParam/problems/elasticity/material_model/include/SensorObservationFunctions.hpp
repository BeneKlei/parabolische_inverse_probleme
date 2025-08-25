#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

#include "ObservationOperatorFactory.hpp"

using namespace dealii;

template <int dim, typename Number>
class SensorObservationFunctions : public Function<dim>
{
public:
    SensorObservationFunctions(ObservationOperatorType observation_operator_type);
    virtual ~SensorObservationFunctions() = default;

    void vector_value(
        const Point<dim, Number> &p,
        Vector<Number> &values
    );
    void vector_value_list(
        const std::vector<Point<dim, Number>> &points,
        std::vector<Vector<Number>> &value_list
    );

    ObservationOperatorType m_observation_operator_type;
    std::vector<Point<dim, Number>> m_included_points;
};
