file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp
)

list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/source/bindings.cpp")

add_library(material_model_operators SHARED ${SOURCES})

target_include_directories(
  material_model_operators
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

PYBIND11_DEAL_II_SETUP_TARGET(material_model_operators)
#install(TARGETS material_model LIBRARY DESTINATION .)

set_target_properties(material_model_operators PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

##################################################################

pybind11_add_module(pymor_dealii_bindings MODULE
  ${CMAKE_CURRENT_SOURCE_DIR}/source/bindings.cpp
)

target_link_libraries(pymor_dealii_bindings PRIVATE material_model_operators)

target_include_directories(pymor_dealii_bindings
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  
)

PYBIND11_DEAL_II_SETUP_TARGET(pymor_dealii_bindings)
install(TARGETS pymor_dealii_bindings LIBRARY DESTINATION .)
