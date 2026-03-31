// =======================================
// ObservationOperatorFactory.cpp  (refactored to Option-1 style)
// =======================================
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_tools.h>

#include "ObservationOperatorFactory.hpp"
#include "utils.hpp"

template class ObservationOperatorFactory<3, double>;

template <int dim, typename Number>
void ObservationOperatorFactory<dim, Number>::assemble_observation(
    const ObservationOperatorFactoryContext<dim, Number>& ctx,
    dealii::SparseMatrix<Number>& observation_operator_matrix,
    dealii::SparsityPattern& observation_operator_sp) const
{
  std::vector<dealii::Point<dim>> sensor_points;

  switch (ctx.observation_operator_type)
  {
    case ObservationOperatorType::Identity:
      assemble_identity_observation(ctx, observation_operator_matrix, observation_operator_sp);
      break;

    case ObservationOperatorType::Boundary:
      assemble_boundary_observation(ctx, observation_operator_matrix, observation_operator_sp);
      break;

    case ObservationOperatorType::Sensors:
      sensor_points = this->_get_sensor_edges(ctx);
      assemble_sensors_observation(ctx, observation_operator_matrix, observation_operator_sp, sensor_points);
      break;

    case ObservationOperatorType::SensorsGrid:
      sensor_points = this->_get_sensor_grids(ctx);
      assemble_sensors_observation(ctx, observation_operator_matrix, observation_operator_sp, sensor_points);
      break;

    default:
      throw std::runtime_error("Unknown ObservationOperatorType.");
  }
}

template <int dim, typename Number>
void ObservationOperatorFactory<dim, Number>::assemble_identity_observation(
    const ObservationOperatorFactoryContext<dim, Number>& ctx,
    dealii::SparseMatrix<Number>& observation_operator_matrix,
    dealii::SparsityPattern& observation_operator_sp) const
{
  observation_operator_sp.copy_from(ctx.space.sparsity_pattern());
  observation_operator_matrix.reinit(observation_operator_sp);
  observation_operator_matrix = 0;

  const auto& dof_handler = ctx.space.dof_handler();
  for (dealii::types::global_dof_index i = 0; i < dof_handler.n_dofs(); ++i)
    observation_operator_matrix.set(i, i, Number(1));
}

template <int dim, typename Number>
void ObservationOperatorFactory<dim, Number>::assemble_boundary_observation(
    const ObservationOperatorFactoryContext<dim, Number>& ctx,
    dealii::SparseMatrix<Number>& observation_operator_matrix,
    dealii::SparsityPattern& observation_operator_sp) const
{
  const ProductFactoryContext<dim, Number> boundary_mass_ctx{
      FEProductType::BoundaryMass,
      ctx.space
  };

  observation_operator_sp.copy_from(ctx.space.sparsity_pattern());
  observation_operator_matrix.reinit(observation_operator_sp);

  m_product_factory.assemble_product(boundary_mass_ctx, observation_operator_matrix);
}

template <int dim, typename Number>
void ObservationOperatorFactory<dim, Number>::assemble_sensors_observation(
    const ObservationOperatorFactoryContext<dim, Number>& ctx,
    dealii::SparseMatrix<Number>& observation_operator_matrix,
    dealii::SparsityPattern& observation_operator_sp,
    std::vector<dealii::Point<dim>> sensor_points) const
{
  const double radius = std::get<double>(ctx.hyperparameter.at("radius"));
  const double tol2   = radius * radius;
  const bool use_boundary_mass_matrix = 
    std::get<bool>(ctx.hyperparameter.at("use_boundary_mass_matrix"));

  dealii::SparseMatrix<Number> G;
  dealii::SparseMatrix<Number> boundary_mass_matrix;

  const ProductFactoryContext<dim, Number> boundary_mass_ctx{
      FEProductType::BoundaryMass,
      ctx.space
  };

  m_product_factory.assemble_product(boundary_mass_ctx, boundary_mass_matrix);

  const auto& dof_handler = ctx.space.dof_handler();
  const unsigned int L = dof_handler.n_dofs();
  const unsigned int l = static_cast<unsigned int>(sensor_points.size());

  std::vector<std::vector<dealii::types::global_dof_index>> rows(l);

  // ----------------------------------------------------
  dealii::MappingQ1<dim> mapping;
  std::vector<dealii::Point<dim>> support_points(dof_handler.n_dofs());
  dealii::DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

  for (unsigned int si = 0; si < l; ++si)
    for (unsigned int i = 0; i < support_points.size(); ++i)
      if ((sensor_points[si] - support_points[i]).norm_square() <= tol2) {
        // std::cout << "Called" << std::endl;
        // std::cout << sensor_points[si] << std::endl;
        // std::cout << support_points[i] << std::endl;
        rows[si].push_back(i);
      }

        

  // ----------------------------------------------------
  dealii::DynamicSparsityPattern dsp(l, L);
  for (unsigned int i = 0; i < l; ++i)
    for (auto dof : rows[i])
      dsp.add(i, dof);

  dealii::SparsityPattern sp_G;
  sp_G.copy_from(dsp);

  G.reinit(sp_G);
  for (unsigned int i = 0; i < l; ++i)
    for (auto dof : rows[i])
      G.set(i, dof, Number(1.0));


  if (use_boundary_mass_matrix) {
    observation_operator_sp.copy_from(utils::make_product_sparsity_AB(G, boundary_mass_matrix));
    observation_operator_matrix.reinit(observation_operator_sp);

    G.mmult(observation_operator_matrix, boundary_mass_matrix);
    return;
  } 
  
  observation_operator_sp.copy_from(G.get_sparsity_pattern()); 
  observation_operator_matrix.reinit(observation_operator_sp);
  observation_operator_matrix.copy_from(G);
  return;
}


template <int dim, typename Number>
std::vector<Point<dim>>
ObservationOperatorFactory<dim, Number>::_get_sensor_edges(
    const ObservationOperatorFactoryContext<dim, Number> &ctx) const
{
    AssertDimension(dim, 3);

    using point_t = Point<dim>;
    std::vector<point_t> sensor_points;

    // --- Domain -------------------------------------------------------------
    const point_t p1 = utils::vec_to_point<dim>(
        std::get<std::vector<double>>(ctx.hyperparameter.at("p1")));
    const point_t p2 = utils::vec_to_point<dim>(
        std::get<std::vector<double>>(ctx.hyperparameter.at("p2")));

    for (unsigned int d = 0; d < dim; ++d)
    {
        AssertThrow(
            p1[d] < p2[d],
            ExcMessage("Invalid cuboid: p1 must be strictly less than p2 in all dimensions"));
    }

    // --- Placement options --------------------------------------------------
    const bool at_top =
        ctx.hyperparameter.count("at_top")
            ? std::get<bool>(ctx.hyperparameter.at("at_top"))
            : true;

    const bool at_bottom =
        ctx.hyperparameter.count("at_bottom")
            ? std::get<bool>(ctx.hyperparameter.at("at_bottom"))
            : true;

    AssertThrow(
        at_top || at_bottom,
        ExcMessage("At least one of 'at_top' or 'at_bottom' must be true"));

    // --- Sensor patch geometry in the yz-plane ------------------------------
    // Full size of the rectangular patch on each x-face where sensors are placed
    const std::vector<double> sensor_patch_size =
        std::get<std::vector<double>>(ctx.hyperparameter.at("sensor_patch_size"));

    AssertThrow(
        sensor_patch_size.size() == 2,
        ExcMessage("sensor_patch_size must have exactly 2 entries: {size_y, size_z}"));

    const double patch_size_y = sensor_patch_size[0];
    const double patch_size_z = sensor_patch_size[1];

    AssertThrow(
        patch_size_y > 0.0 && patch_size_z > 0.0,
        ExcMessage("sensor_patch_size entries must be positive"));

    // Equal spacing along all four edges of the patch
    const double sensor_spacing =
        std::get<double>(ctx.hyperparameter.at("sensor_spacing"));

    AssertThrow(
        sensor_spacing > 0.0,
        ExcMessage("sensor_spacing must be positive"));

    // Optional shift of the patch center in y and z relative to the face center
    const std::vector<double> sensor_patch_center_offset =
        ctx.hyperparameter.count("sensor_patch_center_offset")
            ? std::get<std::vector<double>>(ctx.hyperparameter.at("sensor_patch_center_offset"))
            : std::vector<double>{0.0, 0.0};

    AssertThrow(
        sensor_patch_center_offset.size() == 2,
        ExcMessage("sensor_patch_center_offset must have exactly 2 entries: {offset_y, offset_z}"));

    const double offset_y = sensor_patch_center_offset[0];
    const double offset_z = sensor_patch_center_offset[1];

    // Optional offset from the x-faces inward
    const double x_face_offset =
        ctx.hyperparameter.count("x_face_offset")
            ? std::get<double>(ctx.hyperparameter.at("x_face_offset"))
            : 0.0;

    AssertThrow(
        x_face_offset >= 0.0,
        ExcMessage("x_face_offset must be non-negative"));

    // --- Domain extents -----------------------------------------------------
    const double domain_size_x = p2[0] - p1[0];
    const double domain_size_y = p2[1] - p1[1];
    const double domain_size_z = p2[2] - p1[2];

    AssertThrow(
        patch_size_y <= domain_size_y,
        ExcMessage("sensor_patch_size[0] exceeds domain extent in y"));

    AssertThrow(
        patch_size_z <= domain_size_z,
        ExcMessage("sensor_patch_size[1] exceeds domain extent in z"));

    AssertThrow(
        2.0 * x_face_offset <= domain_size_x,
        ExcMessage("x_face_offset is too large for the x-extent of the domain"));

    // --- Center patch on the yz-face, then apply optional offsets ----------
    const double domain_center_y = 0.5 * (p1[1] + p2[1]);
    const double domain_center_z = 0.5 * (p1[2] + p2[2]);

    const double patch_center_y = domain_center_y + offset_y;
    const double patch_center_z = domain_center_z + offset_z;

    const double y_start = patch_center_y - 0.5 * patch_size_y;
    const double y_end   = patch_center_y + 0.5 * patch_size_y;
    const double z_start = patch_center_z - 0.5 * patch_size_z;
    const double z_end   = patch_center_z + 0.5 * patch_size_z;

    AssertThrow(
        y_start >= p1[1] && y_end <= p2[1],
        ExcMessage("Sensor patch exceeds domain bounds in y"));

    AssertThrow(
        z_start >= p1[2] && z_end <= p2[2],
        ExcMessage("Sensor patch exceeds domain bounds in z"));

    // --- Number of sensors per edge ----------------------------------------
    // Corners are excluded.
    //
    // For example, if patch_size_y = 10 and sensor_spacing = 2:
    // y positions are y_start + 2, y_start + 4, y_start + 6, y_start + 8
    // => 4 interior points on that edge
    //
    // We require exact equal spacing from one corner to the other:
    // patch_size_y / sensor_spacing and patch_size_z / sensor_spacing
    // must both be integers.
    const double n_intervals_y_real = patch_size_y / sensor_spacing;
    const double n_intervals_z_real = patch_size_z / sensor_spacing;

    const double tol = 1e-12;

    const double n_intervals_y_rounded = std::round(n_intervals_y_real);
    const double n_intervals_z_rounded = std::round(n_intervals_z_real);

    AssertThrow(
        std::abs(n_intervals_y_real - n_intervals_y_rounded) < tol,
        ExcMessage("patch_size_y must be an integer multiple of sensor_spacing"));

    AssertThrow(
        std::abs(n_intervals_z_real - n_intervals_z_rounded) < tol,
        ExcMessage("patch_size_z must be an integer multiple of sensor_spacing"));

    const unsigned int n_intervals_y =
        static_cast<unsigned int>(n_intervals_y_rounded);
    const unsigned int n_intervals_z =
        static_cast<unsigned int>(n_intervals_z_rounded);

    AssertThrow(
        n_intervals_y >= 2,
        ExcMessage("patch_size_y must allow at least one interior sensor point"));

    AssertThrow(
        n_intervals_z >= 2,
        ExcMessage("patch_size_z must allow at least one interior sensor point"));

    AssertThrow(
        n_intervals_y == n_intervals_z,
        ExcMessage(
            "Equal spacing with matching edge discretization requires "
            "patch_size_y / sensor_spacing == patch_size_z / sensor_spacing"));

    const unsigned int sensors_per_edge = n_intervals_y - 1;

    // --- Reserve memory -----------------------------------------------------
    const unsigned int n_faces =
        static_cast<unsigned int>(at_top) + static_cast<unsigned int>(at_bottom);

    sensor_points.reserve(4 * sensors_per_edge * n_faces);

    // --- Face positions -----------------------------------------------------
    const double x_top    = p2[0] - x_face_offset;
    const double x_bottom = p1[0] + x_face_offset;

    // --- Helper to add the 4 edges of one rectangular patch ----------------
    auto add_face_edges = [&](const double x_face)
    {
        for (unsigned int i = 1; i <= sensors_per_edge; ++i)
        {
            const double y_pos = y_start + i * sensor_spacing;
            const double z_pos = z_start + i * sensor_spacing;

            // Edge 1: z = z_start, y varies
            sensor_points.emplace_back(point_t(x_face, y_pos, z_start));

            // Edge 2: z = z_end, y varies
            sensor_points.emplace_back(point_t(x_face, y_pos, z_end));

            // Edge 3: y = y_start, z varies
            sensor_points.emplace_back(point_t(x_face, y_start, z_pos));

            // Edge 4: y = y_end, z varies
            sensor_points.emplace_back(point_t(x_face, y_end, z_pos));
        }
    };

    if (at_top)
        add_face_edges(x_top);

    if (at_bottom)
        add_face_edges(x_bottom);

    return sensor_points;
}


template <int dim, typename Number>
std::vector<Point<dim>>
ObservationOperatorFactory<dim, Number>::_get_sensor_grids(
    const ObservationOperatorFactoryContext<dim, Number> &ctx) const
{
    AssertDimension(dim, 3);

    using point_t = Point<dim>;
    std::vector<point_t> sensor_points;

    // --- Domain -------------------------------------------------------------
    const point_t p1 = utils::vec_to_point<dim>(
        std::get<std::vector<double>>(ctx.hyperparameter.at("p1")));
    const point_t p2 = utils::vec_to_point<dim>(
        std::get<std::vector<double>>(ctx.hyperparameter.at("p2")));

    for (unsigned int d = 0; d < dim; ++d)
    {
        AssertThrow(
            p1[d] < p2[d],
            ExcMessage(
                "Invalid cuboid: p1 must be strictly less than p2 in all dimensions"));
    }

    // --- Placement options --------------------------------------------------
    const bool at_top =
        ctx.hyperparameter.count("at_top")
            ? std::get<bool>(ctx.hyperparameter.at("at_top"))
            : true;

    const bool at_bottom =
        ctx.hyperparameter.count("at_bottom")
            ? std::get<bool>(ctx.hyperparameter.at("at_bottom"))
            : true;

    AssertThrow(
        at_top || at_bottom,
        ExcMessage("At least one of 'at_top' or 'at_bottom' must be true"));

    // --- Sensor patch geometry in the yz-plane ------------------------------
    const std::vector<double> sensor_patch_size =
        std::get<std::vector<double>>(ctx.hyperparameter.at("sensor_patch_size"));

    AssertThrow(
        sensor_patch_size.size() == 2,
        ExcMessage("sensor_patch_size must have exactly 2 entries: {size_y, size_z}"));

    const double patch_size_y = sensor_patch_size[0];
    const double patch_size_z = sensor_patch_size[1];

    AssertThrow(
        patch_size_y > 0.0 && patch_size_z > 0.0,
        ExcMessage("sensor_patch_size entries must be positive"));

    // --- Grid resolution on the patch --------------------------------------
    const std::vector<double> grid_sizes =
        std::get<std::vector<double>>(ctx.hyperparameter.at("grid_sizes"));

    AssertThrow(
        grid_sizes.size() == 2,
        ExcMessage("grid_sizes must have exactly 2 entries: {n_y, n_z}"));

    const unsigned int n_y = static_cast<unsigned int>(grid_sizes[0]);
    const unsigned int n_z = static_cast<unsigned int>(grid_sizes[1]);

    AssertThrow(n_y >= 1, ExcMessage("grid_sizes[0] must be at least 1"));
    AssertThrow(n_z >= 1, ExcMessage("grid_sizes[1] must be at least 1"));

    // --- Optional shift of the patch center in y and z ----------------------
    const std::vector<double> sensor_patch_center_offset =
        ctx.hyperparameter.count("sensor_patch_center_offset")
            ? std::get<std::vector<double>>(ctx.hyperparameter.at("sensor_patch_center_offset"))
            : std::vector<double>{0.0, 0.0};

    AssertThrow(
        sensor_patch_center_offset.size() == 2,
        ExcMessage(
            "sensor_patch_center_offset must have exactly 2 entries: {offset_y, offset_z}"));

    const double offset_y = sensor_patch_center_offset[0];
    const double offset_z = sensor_patch_center_offset[1];

    // --- Optional offset from the x-faces inward ----------------------------
    const double x_face_offset =
        ctx.hyperparameter.count("x_face_offset")
            ? std::get<double>(ctx.hyperparameter.at("x_face_offset"))
            : 0.0;

    AssertThrow(
        x_face_offset >= 0.0,
        ExcMessage("x_face_offset must be non-negative"));

    // --- Domain extents -----------------------------------------------------
    const double domain_size_x = p2[0] - p1[0];
    const double domain_size_y = p2[1] - p1[1];
    const double domain_size_z = p2[2] - p1[2];

    AssertThrow(
        patch_size_y <= domain_size_y,
        ExcMessage("sensor_patch_size[0] exceeds domain extent in y"));

    AssertThrow(
        patch_size_z <= domain_size_z,
        ExcMessage("sensor_patch_size[1] exceeds domain extent in z"));

    AssertThrow(
        2.0 * x_face_offset <= domain_size_x,
        ExcMessage("x_face_offset is too large for the x-extent of the domain"));

    // --- Center patch on the yz-face, then apply optional offsets ----------
    const double domain_center_y = 0.5 * (p1[1] + p2[1]);
    const double domain_center_z = 0.5 * (p1[2] + p2[2]);

    const double patch_center_y = domain_center_y + offset_y;
    const double patch_center_z = domain_center_z + offset_z;

    const double y_start = patch_center_y - 0.5 * patch_size_y;
    const double y_end   = patch_center_y + 0.5 * patch_size_y;
    const double z_start = patch_center_z - 0.5 * patch_size_z;
    const double z_end   = patch_center_z + 0.5 * patch_size_z;

    AssertThrow(
        y_start >= p1[1] && y_end <= p2[1],
        ExcMessage("Sensor patch exceeds domain bounds in y"));

    AssertThrow(
        z_start >= p1[2] && z_end <= p2[2],
        ExcMessage("Sensor patch exceeds domain bounds in z"));

    // --- Grid spacing on the patch -----------------------------------------
    const double dy = (n_y > 1) ? (y_end - y_start) / (n_y - 1) : 0.0;
    const double dz = (n_z > 1) ? (z_end - z_start) / (n_z - 1) : 0.0;

    // --- Reserve memory -----------------------------------------------------
    const unsigned int n_faces =
        static_cast<unsigned int>(at_top) + static_cast<unsigned int>(at_bottom);

    sensor_points.reserve(
        static_cast<std::size_t>(n_faces) *
        static_cast<std::size_t>(n_y) *
        static_cast<std::size_t>(n_z));

    // --- Face positions -----------------------------------------------------
    const double x_top    = p2[0] - x_face_offset;
    const double x_bottom = p1[0] + x_face_offset;

    // --- Helper to add one full rectangular grid on a face -----------------
    auto add_face_grid = [&](const double x_face)
    {
        for (unsigned int iy = 0; iy < n_y; ++iy)
        {
            const double y_pos =
                (n_y == 1) ? patch_center_y : y_start + iy * dy;

            for (unsigned int iz = 0; iz < n_z; ++iz)
            {
                const double z_pos =
                    (n_z == 1) ? patch_center_z : z_start + iz * dz;

                sensor_points.emplace_back(point_t(x_face, y_pos, z_pos));
            }
        }
    };

    if (at_top)
        add_face_grid(x_top);

    if (at_bottom)
        add_face_grid(x_bottom);

    return sensor_points;
}


// template <int dim, typename Number>
// std::vector<Point<dim>> ObservationOperatorFactory<dim, Number>::_get_sensor_grids(
//     const ObservationOperatorFactoryContext<dim, Number>& ctx
// ) const
// template <int dim, typename Number>
// std::vector<Point<dim>> ObservationOperatorFactory<dim, Number>::_get_sensor_grids(
//     const ObservationOperatorFactoryContext<dim, Number>& ctx
// ) const
// {   
//     AssertDimension(dim, 3);
//     std::vector<Point<dim>> sensor_points;
//     size_t idx;

//     double x_coord;
//     double y_coord;
//     double z_coord;

//     //std::vector<double> spatial_resolution = std::get<std::vector<double>>(ctx.hyperparameter.at("spatial_resolution"));
//     std::vector<double> grid_sizes = std::get<std::vector<double>>(ctx.hyperparameter.at("grid_sizes"));
    
//     sensor_points.resize(grid_sizes[0] * grid_sizes[1] * grid_sizes[2]);    
//     idx = 0;

//     for (unsigned int i = 0; i < grid_sizes[0]; i++) {
//         x_coord = -0.1 + 0 + i * (0.2 / (grid_sizes[0] - 1));
        
//         for (unsigned int j = 0; j < grid_sizes[1]; j++) {
//             y_coord = -15.0 + 1 + j * (28.0 / (grid_sizes[1] - 1));
//             //y_coord = -15.0 + j * (30.0 / (grid_sizes[1] - 1));
            
//             for (unsigned int k = 0; k < grid_sizes[2]; k++) {
//                 z_coord = -15.0 + 1 + k * (28.0 / (grid_sizes[2] - 1));
//                 //z_coord = -15.0 + k * (30.0 / (grid_sizes[2] - 1));
                
//                 sensor_points[idx] = Point<3>(x_coord, y_coord, z_coord);
//                 idx++;

//                 // std::cout << "----------------------------------------" << std::endl;
//                 // std::cout << x_coord << std::endl;
//                 // std::cout << y_coord << std::endl;
//                 // std::cout << z_coord << std::endl;
//             }
//         }
//     }   

//     return sensor_points;
// }

