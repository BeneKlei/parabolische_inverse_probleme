// #include "BodyForce.hpp"

// void BodyForce::vector_value_list(const std::vector<Point<3>> &points, std::vector<Vector<double>> &value_list) const
// {
//     Assert (value_list.size() == points.size(),
//     ExcDimensionMismatch (value_list.size(), points.size()));

//     const unsigned int n_points = points.size();

//     for (unsigned int p=0; p<n_points; ++p)
//         this->vector_value(points[p], value_list[p]);
// }

// void CenterExciteBodyForce::vector_value(const Point<3> &p, Vector<double> &values) const 
// {
//     double fx, fy, fz, ft;
//     // ---------------------- ft ----------------------
//     if (get_time() <= 0.5)  {
//         if (get_time() <= 0) {
//             ft = 0;
//         } else {
//             ft = 1.0 * get_time();
//         }
//     } else {
//         ft = 0;
//     }
//     // ---------------------- fx  ----------------------
//     fx = 1;
//     // ---------------------- fy ----------------------
//     if (p(1) <= 1) {
//         if (p(1) <= 0) {
//             if (p(1) <= -1) {
//                 fy = 0;
//             } else {
//                 fy = p(1) + 1;
//             }
//         } else {
//             fy = -p(1) + 1; 
//         }
//     } else {
//         fy = 0;
//     }
//     // ---------------------- fz  ----------------------
//     if (p(2) <= 1) {
//         if (p(2) <= 0) {
//             if (p(2) <= -1) {
//                 fz = 0;
//             } else {
//                 fz = p(2) + 1;
//             }
//         } else {
//             fz = -p(2) + 1; 
//         }
//     } else {
//         fz = 0;
//     }
//     values(0) = 0;
//     values(1) = 0;
//     values(2) = ft*fx*fy*fz;
// }

