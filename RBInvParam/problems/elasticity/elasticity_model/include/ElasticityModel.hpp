#include <operators/BilinearOperator.hpp>
#include <MaterialModel.hpp>
#include <MaterialMatricesFactory.hpp>

class ElasticityModel : public MaterialModel
{
public:
    using Op = BilinearAqOp<Number>;
    // TODO Use ElasticityModelConfig
    explicit ElasticityModel(const MaterialModelConfig& config);
    
    void assemble_system_operator(Vector<Number> q);

private:
    std::shared_ptr<const MatrixStack<Number>> m_matrix_stack;
    std::unique_ptr<Op> m_op;

    MaterialMatricesFactory<dim, Number> m_material_matrices_factory = MaterialMatricesFactory<3, Number>();
};

