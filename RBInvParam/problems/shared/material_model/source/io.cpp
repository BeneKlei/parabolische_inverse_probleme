#include <deal.II/numerics/matrix_tools.h>

#include <sys/stat.h>
#include <fstream>
#include <iostream>

#include "../include/io.hpp"



// TODO Use modern file management, move to MaterialModel
// void writeVtk(Vector<double> v, const char *filename, const DoFHandler<3> &dof_handler) {
//     DataOut<3> data_out;
//     data_out.attach_dof_handler(dof_handler);
//     std::vector<std::string> solution_names;

//     solution_names.push_back("x");
//     solution_names.push_back("y");
//     solution_names.push_back("z");

//     std::vector<DataComponentInterpretation::DataComponentInterpretation> dci(3);
//     for (unsigned int i=0;i<3;i++)
//     dci[i] = DataComponentInterpretation::component_is_part_of_vector;
//     data_out.add_data_vector(v, solution_names, DataOut<3>::type_dof_data ,dci);
//     data_out.add_data_vector(v, solution_names);
//     data_out.build_patches();

//     std::ofstream output(filename);
//     data_out.write_vtk(output);
//     output.close();
// }

void write_matrix(const FullMatrix<double> &m, std::filesystem::path path)
{
    struct stat finfo;
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Unable to open file " << path << std::endl;
        return;
    }

    for (unsigned int i = 0; i < m.m(); i++)
    {
        for (unsigned int j = 0; j < m.n(); j++)
        {
            out << m[i][j] << " ";
        }
        out << std::endl;
    }
    out.close();
    int ret = stat(path.c_str(), &finfo);
    if (ret != 0)
        std::cerr << "A problem writing into file " << path << " occurred." << std::endl;
}


void write_matrix(const SparseMatrix<double> &m, std::filesystem::path path)
{   
    FullMatrix<double> full_mat(m.m(), m.n());
        // Manually copy non-zero values from the SparseMatrix to FullMatrix
    for (unsigned int i = 0; i < m.m(); ++i)
    {
        for (typename SparseMatrix<double>::const_iterator iter = m.begin(i); iter != m.end(i); ++iter)
        {
            full_mat(i, iter->column()) = iter->value();
        }
    }

    write_matrix(full_mat, path);
}