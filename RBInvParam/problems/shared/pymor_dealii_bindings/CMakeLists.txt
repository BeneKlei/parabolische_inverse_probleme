file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ./source/*.cpp
)

pybind11_add_module(
  pymor_dealii_bindings 
  ${SOURCES}
)
target_include_directories(
  pymor_dealii_bindings
  PRIVATE 
  ./include
)

PYBIND11_DEAL_II_SETUP_TARGET(pymor_dealii_bindings)
install(TARGETS pymor_dealii_bindings LIBRARY DESTINATION .)
