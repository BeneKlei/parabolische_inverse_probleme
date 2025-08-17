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
    int par_dim = 2;
    int nt = 50;
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
  void assemble_system_matrix_derivative(
    FullMatrix<Number>& system_matrix_derivative,
    const Vector<Number>& state_DoFs
  );

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
  void assemble_euclidian_matrix(SparseMatrix<Number>& operator_matrix);

  void assemble_mass_matrix(SparseMatrix<Number>& mass_matrix);
  void assemble_observation_operator_matrix(SparseMatrix<Number>& operator_matrix, std::string operator_name);
  

  void assemble_bilinear_cost_matrix(
    SparseMatrix<Number>& matrix,
    const SparseMatrix<Number>& prod_C,
    const SparseMatrix<Number>& C
  );
  void clear_rhs_boundary_dofs(Vector<Number>& v);
  
  void output_results(Vector<double>& solution) const;
    
  const SparsityPattern& sparsity_pattern() const { return m_sparsity_pattern; }
  uint32_t n_dofs() const { return m_dof_handler.n_dofs(); }
  const std::vector<Vector<Number>>& get_force_list() const { return m_force_list; };

  size_t m_param_space_dim;
  size_t m_state_space_dim;
  

private:
  const MaterialModelConfig m_config;
  Triangulation<dim> m_triangulation;
  FESystem<dim> m_fe;
  DoFHandler<dim> m_dof_handler;
  SparsityPattern m_sparsity_pattern;
  SparsityPattern m_bilinear_cost_sparsity_pattern;

  AffineConstraints<Number> m_BC_constraints;
  SparseILU<Number> m_solver;

  BodyForce m_body_force;

  MatrixStack m_system_matricies;
  MatrixStack m_adjoint_system_matricies;

  std::vector<Vector<Number>> m_force_list;

  void setup_system_matricies();
  void setup_adjoint_system_matricies();
  void setup_BC_constraints();
  void assemble_force_list();
  void assemble_force(Vector<Number>& result, double time);
};


