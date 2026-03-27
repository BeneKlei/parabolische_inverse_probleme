#include <cassert>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/component_mask.h>

#include "BodyForceFactory.hpp"


void BodyForce::vector_value_list(const std::vector<Point<3>> &points, std::vector<Vector<double>> &value_list) const
{
    Assert (value_list.size() == points.size(),
    ExcDimensionMismatch (value_list.size(), points.size()));

    const unsigned int n_points = points.size();

    for (unsigned int p=0; p<n_points; ++p)
        this->vector_value(points[p], value_list[p]);
}

void CenterExciteBodyForce::vector_value(const Point<3> &p, Vector<double> &values) const 
{
    double fx, fy, fz, ft;
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
    // ---------------------- fx  ----------------------
    fx = 1;
    // ---------------------- fy ----------------------
    if (p(1) <= 1) {
        if (p(1) <= 0) {
            if (p(1) <= -1) {
                fy = 0;
            } else {
                fy = p(1) + 1;
            }
        } else {
            fy = -p(1) + 1; 
        }
    } else {
        fy = 0;
    }
    // ---------------------- fz  ----------------------
    if (p(2) <= 1) {
        if (p(2) <= 0) {
            if (p(2) <= -1) {
                fz = 0;
            } else {
                fz = p(2) + 1;
            }
        } else {
            fz = -p(2) + 1; 
        }
    } else {
        fz = 0;
    }
    // values(0) = 0;
    // values(1) = 0;
    // values(2) = ft*fx*fy*fz;

    
    values(0) = ft*fx*fy*fz;
    values(1) = 0;
    values(2) = 0;
}


void CenterExciteWaveBodyForce::vector_value(const Point<3> &p, Vector<double> &values) const 
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

    const double t = this->get_time();
    double pulse_t = 0.0;

    if (std::abs(t - 8.2e-6) <= end_time) {
        pulse_t =
            std::sin(4.0e5 * (t - 7.0e-6)) *
            std::exp(-0.06 * std::pow(4.0e5 * (t - 8.2e-6), 2.0));
    } else {
        pulse_t = 0.0;
    }

    const double yy = p(1) - 0.15;
    const double zz = p(2) - 0.15;
    const double spatial_decay = std::exp(-700.0 * (yy * yy + zz * zz));

    const double excite_x = pulse_t * (-1.0) * spatial_decay;
    const double excite_y = pulse_t * yy     * spatial_decay;
    const double excite_z = pulse_t * zz     * spatial_decay;

    // =========================================================
    // Total body force = first excitation + second excitation
    // =========================================================
    values.reinit(3);
    values(0) = excite_x;
    values(1) = excite_y;
    values(2) = excite_z;
}

void GaussianBodyForce::vector_value(const Point<3> &p, Vector<double> &values) const 
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

    const double r = p.distance(center);
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
  case BodyForceType::CenterExcite:
    std::cout << "\t Using CenterExcite BodyForce" << std::endl;
    return BodyForceFactory::assemble_center_excite_body_force(
        ctx
    );
    break;
  case BodyForceType::Gaussian:
    std::cout << "\t Using Gaussian BodyForce" << std::endl;
    return BodyForceFactory::assemble_gaussian_body_force(
        ctx
    );
    break;
  default:
    throw std::runtime_error("Unknown BodyForceType.");
  }
}

template <int dim, typename Number>
std::unique_ptr<BodyForce> BodyForceFactory<dim, Number>::assemble_center_excite_body_force(
  const BodyForceFactoryContext<dim, Number>& ctx) const
{
    double end_time = std::get<double>(ctx.hyperparameter.at("end_time"));
    double factor = std::get<double>(ctx.hyperparameter.at("factor"));
    return std::make_unique<CenterExciteBodyForce>(
        end_time,
        factor
    );
};

template <int dim, typename Number>
std::unique_ptr<BodyForce> BodyForceFactory<dim, Number>::assemble_gaussian_body_force(
  const BodyForceFactoryContext<dim, Number>& ctx) const
{
  //check_required_keys<double>(ctx.hyperparameter, {"center", "width"});
  std::vector<double> center = std::get<std::vector<double>>(ctx.hyperparameter.at("center"));
  double sigma = std::get<double>(ctx.hyperparameter.at("sigma"));
  double end_time = std::get<double>(ctx.hyperparameter.at("end_time"));

  return std::make_unique<GaussianBodyForce>(
      Point<3>(0,0,0), 
      sigma,
      end_time
  );
  
};


