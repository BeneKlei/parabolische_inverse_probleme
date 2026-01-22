file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp
)
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/source/bindings.cpp")

add_library(material_model_core SHARED ${SOURCES})

target_include_directories(
  material_model_core
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
  material_model_core 
  PUBLIC 
  material_model_operators
)

PYBIND11_DEAL_II_SETUP_TARGET(material_model_core)
#install(TARGETS material_model LIBRARY DESTINATION .)

set_target_properties(material_model_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

##################################################################

pybind11_add_module(material_model MODULE
  ${CMAKE_CURRENT_SOURCE_DIR}/source/bindings.cpp
)

target_link_libraries(material_model PRIVATE material_model_core)

target_include_directories(material_model
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
)

PYBIND11_DEAL_II_SETUP_TARGET(material_model)

set_target_properties(material_model PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
