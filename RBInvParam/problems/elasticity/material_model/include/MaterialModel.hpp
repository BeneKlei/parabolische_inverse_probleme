#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h> 
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>


#include "MatrixStack.hpp"
#include "BodyForce.hpp"


using namespace dealii;

typedef double Number;
// TODO make Class for this with "highlevel" pymor like interface
typedef std::vector<dealii::Vector<Number>> VectorArray;

struct MaterialModelConfig {
    double T_initial = 0.0;
    double T_final = 1.0;
    double delta_t = 1.0 / 50;
    int polynomial_degree = 1;
    int par_dim = 2;
    int refine_global = 2;
};

class MaterialModel
{
public:
  static constexpr size_t dim{3};

  explicit MaterialModel(const MaterialModelConfig& config);
  virtual ~MaterialModel() = default;

  void make_grid();
  void setup_system();

  Vector<Number> m_q;
  Vector<Number> m_d;

  void _solve();

  void assemble_system_matrix(SparseMatrix<Number>& system_matrix);

  template <typename Integrand>
  void _assemble_product_matrix(SparseMatrix<Number>& matrix,
                                Integrand integrand,
                                std::optional<std::reference_wrapper<const AffineConstraints<Number>>> constraints = std::nullopt);

  void assemble_l2_matrix(SparseMatrix<Number>& l2_matrix);
  void assemble_l2_0_matrix(SparseMatrix<Number>& l2_0_matrix);
  void assemble_h1_semi_matrix(SparseMatrix<Number>& h1_semi_matrix);
  void assemble_h1_0_semi_matrix(SparseMatrix<Number>& h1_0_semi_matrix);
  void assemble_h1_matrix(SparseMatrix<Number>& h1_matrix);
  void assemble_h1_0_matrix(SparseMatrix<Number>& h1_0_matrix);

  void assemble_mass_matrix(SparseMatrix<Number>& mass_matrix);
  
  void output_results(Vector<double>& solution) const;
    
  const SparsityPattern& sparsity_pattern() const { return m_sparsity_pattern; }
  uint32_t n_dofs() const { return m_dof_handler.n_dofs(); }
  

private:
  const MaterialModelConfig m_config;
  Triangulation<dim> m_triangulation;
  FESystem<dim> m_fe;
  DoFHandler<dim> m_dof_handler;
  SparsityPattern m_sparsity_pattern;

  AffineConstraints<Number> m_BC_constraints;
  SparseILU<Number> m_solver;

  BodyForce m_body_force;

  uint32_t m_K;

  SparseMatrix<Number> m_lhs;
  Vector<Number> m_rhs;
  Vector<Number> m_solution;

  MatrixStack m_system_matricies;
  MatrixStack m_adjoint_system_matricies;

  SparseMatrix<Number> m_system_matrix;  
  std::vector<Vector<Number>> m_L;

  void setup_system_matricies();
  void setup_adjoint_system_matricies();
  void setup_BC_constraints();
  void assemble_force_list();
  void assemble_force(Vector<Number>& result, double time);
};


