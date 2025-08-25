#include "SensorObservationFunctions.hpp"

template class SensorObservationFunctions<3, double>;

template <int dim, typename Number>
SensorObservationFunctions<dim, Number>::SensorObservationFunctions(ObservationOperatorType observation_operator_type) : 
    Function<3>(3), 
    m_observation_operator_type(observation_operator_type) 
{
    AssertDimension(dim, 3);
    switch (m_observation_operator_type) {
    case ObservationOperatorType::SensorsR9d:
        m_included_points.reserve(64);
        for (unsigned int i = 0; i < 8; i++) {
            m_included_points[i] = Point<3>(0.1, -16.0 + i*4.0, -16.0);
            m_included_points[i + 8] = Point<3>(0.1, 16.0, -16.0 + i*4.0);
            m_included_points[i + 16] = Point<3>(0.1, 16.0 - i*4.0, 16.0);
            m_included_points[i + 24] = Point<3>(0.1, -16.0, 16.0 - i*4.0);
            m_included_points[i + 32] = Point<3>(-0.1, -16.0 + i*4.0, -16.0);
            m_included_points[i + 40] = Point<3>(-0.1, 16.0, -16.0 + i*4.0);
            m_included_points[i + 48] = Point<3>(-0.1, 16.0 - i*4.0, 16.0);
            m_included_points[i + 56] = Point<3>(-0.1, -16.0, 16.0 - i*4.0);
        }
        break;
   case ObservationOperatorType::SensorsR8d:
        m_included_points.reserve(56);
        for (unsigned int i = 0; i < 7; i++) {
            m_included_points[i] = Point<3>(0.1, -14.0 + i*4.0, -14.0);
            m_included_points[i + 7] = Point<3>(0.1, 14.0, -14.0 + i*4.0);
            m_included_points[i + 14] = Point<3>(0.1, 14.0 - i*4.0, 14.0);
            m_included_points[i + 21] = Point<3>(0.1, -14.0, 14.0 - i*4.0);
            m_included_points[i + 28] = Point<3>(-0.1, -14.0 + i*4.0, -14.0);
            m_included_points[i + 35] = Point<3>(-0.1, 14.0, -14.0 + i*4.0);
            m_included_points[i + 42] = Point<3>(-0.1, 14.0 - i*4.0, 14.0);
            m_included_points[i + 49] = Point<3>(-0.1, -14.0, 14.0 - i*4.0);
        }
        break;
    default:
        throw std::runtime_error(
            "ObservationOperatorType is unknown."
        );
    }
};


template <int dim, typename Number>
void SensorObservationFunctions<dim, Number>::vector_value(
    const Point<dim, Number> &p,
    Vector<Number> &values)
{
    AssertDimension(values.size(), 3);
    values = 0;

    const Number tol = Number(1e-10);

    for (const Point<dim, Number> &p_sensor : m_included_points)
    {
        if (p.distance(p_sensor) <= tol)
        {
            values[0] = Number(1);
            values[1] = Number(1);
            values[2] = Number(1);
            break;
        }
    }
}

template <int dim, typename Number>
void SensorObservationFunctions<dim, Number>::vector_value_list(
    const std::vector<Point<dim, Number>> &points,
    std::vector<Vector<Number>> &value_list)
{
    Assert (value_list.size() == points.size(),
    ExcDimensionMismatch (value_list.size(), points.size()));

    const unsigned int n_points = points.size();

    for (unsigned int p=0; p<n_points; ++p)
        this->vector_value(points[p], value_list[p]);
}