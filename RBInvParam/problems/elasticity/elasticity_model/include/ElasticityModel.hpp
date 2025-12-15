#include <MaterialModel.hpp>

class ElasticityModel : public MaterialModel
{
public:
    ElasticityModel(const MaterialModelConfig& config) : MaterialModel(config) {}
    void test();
};

