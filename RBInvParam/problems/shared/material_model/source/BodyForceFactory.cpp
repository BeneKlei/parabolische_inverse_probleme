#include <cassert>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/component_mask.h>

#include "BodyForceFactory.hpp"
#include "utils.hpp"

void BodyForce::vector_value_list(const std::vector<Point<3>> &points, std::vector<Vector<double>> &value_list) const
{
    Assert (value_list.size() == points.size(),
    ExcDimensionMismatch (value_list.size(), points.size()));

    const unsigned int n_points = points.size();

    for (unsigned int p=0; p<n_points; ++p)
        this->vector_value(points[p], value_list[p]);
}

void SharpPulseBodyForce::vector_value(const Point<3> &p, Vector<double> &values) const 
{
    double fx, fy, fz, ft;

    const double yy = p(1) - origin(1);
    const double zz = p(2) - origin(2);

    // ---------------------- ft ----------------------
    if (get_time() <= end_time)  {
        if (get_time() <= 0) {
            ft = 0;
        } else {
            ft = factor * get_time();
        }
    } else {
        ft = 0;
    }

    // ---------------------- fx ----------------------
    fx = 1;

    // ---------------------- fy ----------------------
    if (yy <= width) {
        if (yy <= 0) {
            if (yy <= -width) {
                fy = 0;
            } else {
                fy = yy + 1;
            }
        } else {
            fy = -yy + 1; 
        }
    } else {
        fy = 0;
    }

    // ---------------------- fz ----------------------
    if (zz <= width) {
        if (zz <= 0) {
            if (zz <= -width) {
                fz = 0;
            } else {
                fz = zz + 1;
            }
        } else {
            fz = -zz + 1; 
        }
    } else {
        fz = 0;
    }

    values.reinit(3);
    values(0) = ft * fx * fy * fz;
    values(1) = 0;
    values(2) = 0;
}

void WavePulseBodyForce::vector_value(const Point<3> &p, Vector<double> &values) const 
{
    // =========================================================
    //
    // Original:
    //   f(t,x) = p(t) s(x)
    //
    //   p(t) = sin(4e5 (t - 7e-6)) * exp(-0.06 * (4e5 (t - 8.2e-6))^2)
    //          if |t - 8.2e-6| < 4e-5, else 0
    //
    //   s(x) = [ x1-0.15, x2-0.15, -1 ]^T
    //          * exp(-700((x1-0.15)^2 + (x2-0.15)^2))
    //
    // Axis swap requested:
    //   z -> x, x -> y, y -> z
    //
    // After swap in deal.II coordinates p(0)=x, p(1)=y, p(2)=z:
    //   s_swapped(p) = [ -1, p(1)-0.15, p(2)-0.15 ]^T
    //                  * exp(-700((p(1)-0.15)^2 + (p(2)-0.15)^2))
    // =========================================================

    const double t = this->get_time() * time_scaling_factor;
    double pulse_t = 0.0;

    if (std::abs(t - 8.2e-6) <= end_time) {
        pulse_t =
            std::sin(4.0e5 * (t - 7.0e-6)) *
            std::exp(-0.06 * std::pow(4.0e5 * (t - 8.2e-6), 2.0));
    } else {
        pulse_t = 0.0;
    }

    const double yy = p(1) - origin(1);
    const double zz = p(2) - origin(2);

    const double spatial_decay = std::exp(-700.0 * (yy * yy + zz * zz));

    const double excite_x = pulse_t * (-1.0) * spatial_decay;
    const double excite_y = pulse_t * yy     * spatial_decay;
    const double excite_z = pulse_t * zz     * spatial_decay;

    // =========================================================
    // Total body force = first excitation + second excitation
    // =========================================================
    values.reinit(3);
    values(0) = factor * excite_x;
    values(1) = factor * excite_y;
    values(2) = factor * excite_z;
}

void GaussianPulseBodyForce::vector_value(const Point<3> &p, Vector<double> &values) const 
{
    double amplitude;
    if (get_time() <= end_time)  {
        if (get_time() <= 0) {
            amplitude = 0;
        } else {
            amplitude = (1.0 / end_time) * get_time();
        }
    } else {
        amplitude = 0;
    }

    const double r = p.distance(origin);
    const double gaussian = amplitude * std::exp(-(r*r)/(2.0*width*width));
    
    values(0) = gaussian;
    values(1) = 0;
    values(2) = 0;
};

// ---------------------------------------------------------------------------------------------------------------------
template class BodyForceFactory<3, double>;

template <int dim, typename Number>
std::unique_ptr<BodyForce>  BodyForceFactory<dim, Number>::assemble_body_force(
  const BodyForceFactoryContext<dim, Number>& ctx) const
{
  switch (ctx.body_force_type)
  {
  case BodyForceType::SharpPulse:
    std::cout << "\t Using SharpPulse BodyForce" << std::endl;
    return BodyForceFactory::assemble_sharp_pulse_body_force(
        ctx
    );
    break;
  case BodyForceType::WavePulse:
    std::cout << "\t Using WavePulse BodyForce" << std::endl;
    return BodyForceFactory::assemble_wave_pulse_body_force(
        ctx
    );
    break;
  case BodyForceType::GaussianPulse:
    std::cout << "\t Using GaussianPulse BodyForce" << std::endl;
    return BodyForceFactory::assemble_gaussian_pulse_body_force(
        ctx
    );
    break;
  default:
    throw std::runtime_error("Unknown BodyForceType.");
  }
}

template <int dim, typename Number>
std::unique_ptr<BodyForce> BodyForceFactory<dim, Number>::assemble_sharp_pulse_body_force(
  const BodyForceFactoryContext<dim, Number>& ctx) const
{
    const Point<dim> origin = utils::vec_to_point<dim>(
        std::get<std::vector<double>>(ctx.hyperparameter.at("origin"))
    );

    double end_time = std::get<double>(ctx.hyperparameter.at("end_time"));
    double factor = std::get<double>(ctx.hyperparameter.at("factor"));
    double width = std::get<double>(ctx.hyperparameter.at("width"));
    return std::make_unique<SharpPulseBodyForce>(
        origin,
        end_time,
        factor,
        width
    );
};

template <int dim, typename Number>
std::unique_ptr<BodyForce> BodyForceFactory<dim, Number>::assemble_wave_pulse_body_force(
  const BodyForceFactoryContext<dim, Number>& ctx) const
{   
    const Point<dim> origin = utils::vec_to_point<dim>(
        std::get<std::vector<double>>(ctx.hyperparameter.at("origin"))
    );

    double end_time = std::get<double>(ctx.hyperparameter.at("end_time"));
    double factor = std::get<double>(ctx.hyperparameter.at("factor"));
    double time_scaling_factor = std::get<double>(ctx.hyperparameter.at("time_scaling_factor"));

    return std::make_unique<WavePulseBodyForce>(
        origin,
        end_time,
        factor,
        time_scaling_factor
    );
};

template <int dim, typename Number>
std::unique_ptr<BodyForce> BodyForceFactory<dim, Number>::assemble_gaussian_pulse_body_force(
  const BodyForceFactoryContext<dim, Number>& ctx) const
{
    const Point<dim> origin = utils::vec_to_point<dim>(
        std::get<std::vector<double>>(ctx.hyperparameter.at("origin"))
    );

    double sigma = std::get<double>(ctx.hyperparameter.at("sigma"));
    double end_time = std::get<double>(ctx.hyperparameter.at("end_time"));

    return std::make_unique<GaussianPulseBodyForce>(
        origin, 
        sigma,
        end_time
    );
  
};


