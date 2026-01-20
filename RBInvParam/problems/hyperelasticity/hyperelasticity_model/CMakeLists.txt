file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ./source/*.cpp
)

pybind11_add_module(
    hyper_elasticity_model
    ${SOURCES}
)
target_include_directories(
  hyper_elasticity_model
  PRIVATE 
  ./include
)

target_link_libraries(
    hyper_elasticity_model 
    PRIVATE 
    material_model_core
)


PYBIND11_DEAL_II_SETUP_TARGET(hyper_elasticity_model)
#install(TARGETS material_model LIBRARY DESTINATION .)

set_target_properties(hyper_elasticity_model PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)


