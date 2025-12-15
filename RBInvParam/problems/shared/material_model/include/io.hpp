#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>

#include <filesystem>

using namespace dealii;

//void writeVtk(Vector<double> v, const char *filename, const DoFHandler<3> &dof_handler);
void write_matrix(const FullMatrix<double> &m, std::filesystem::path path);
void write_matrix(const SparseMatrix<double> &m, std::filesystem::path path);
