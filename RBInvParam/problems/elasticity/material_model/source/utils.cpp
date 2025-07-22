// #include <deal.II/lac/sparse_matrix.h>
// #include <cmath>

// #include "utils.hpp"

// bool is_symmetric(const SparseMatrix<double> &matrix, const double tolerance)
// {
//     Assert(matrix.m() == matrix.n(), dealii::ExcNotQuadratic());

//     for (unsigned int i = 0; i < matrix.m(); ++i)
//     {
//         for (dealii::SparseMatrix<double>::const_iterator it = matrix.begin(i);
//              it != matrix.end(i); ++it)
//         {
//             const unsigned int j = it->column();
//             if (j < i)
//                 continue; // already checked lower triangle

//             double a_ij = it->value();
//             double a_ji = matrix.el(j, i);

//             if (std::abs(a_ij - a_ji) > tolerance)
//                 return false;
//         }
//     }
//     return true;
// }
