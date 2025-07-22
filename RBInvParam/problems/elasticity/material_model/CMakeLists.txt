file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ./source/*.cpp
)
list(REMOVE_ITEM SOURCES "./source/main.cpp")
list(REMOVE_ITEM SOURCES "./source/bindings.cpp")

pybind11_add_module(
    material_model
    ${SOURCES}
)
target_include_directories(
  material_model
  PRIVATE 
  ./include
)

PYBIND11_DEAL_II_SETUP_TARGET(material_model)
#install(TARGETS material_model LIBRARY DESTINATION .)

set_target_properties(material_model PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
