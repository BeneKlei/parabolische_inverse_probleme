file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ./source/*.cpp
)

pybind11_add_module(
    hyperelasticity_model
    ${SOURCES}
)
target_include_directories(
  hyperelasticity_model
  PRIVATE 
  ./include
)

target_link_libraries(
    hyperelasticity_model 
    PRIVATE 
    material_model_core
)


PYBIND11_DEAL_II_SETUP_TARGET(hyperelasticity_model)
#install(TARGETS material_model LIBRARY DESTINATION .)

set_target_properties(hyperelasticity_model PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)


