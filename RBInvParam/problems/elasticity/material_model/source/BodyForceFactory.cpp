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
    if (get_time() <= 0.5)  {
        if (get_time() <= 0) {
            ft = 0;
        } else {
            ft = 1.0 * get_time();
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

void GaussianBodyForce::vector_value(const Point<3> &p, Vector<double> &values) const 
{
    double amplitude;
    if (get_time() <= 0.5)  {
        if (get_time() <= 0) {
            amplitude = 0;
        } else {
            amplitude = 1.0 * get_time();
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
  return std::make_unique<CenterExciteBodyForce>();
};

template <int dim, typename Number>
std::unique_ptr<BodyForce> BodyForceFactory<dim, Number>::assemble_gaussian_body_force(
  const BodyForceFactoryContext<dim, Number>& ctx) const
{
  //check_required_keys<double>(ctx.hyperparameter, {"center", "width"});
  std::vector<double> center = std::get<std::vector<double>>(ctx.hyperparameter.at("center"));
  double sigma = std::get<double>(ctx.hyperparameter.at("sigma"));

  return std::make_unique<GaussianBodyForce>(
      Point<3>(0,0,0), sigma
  );
  
};


