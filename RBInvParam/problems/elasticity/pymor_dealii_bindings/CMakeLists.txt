pybind11_add_module(
  pymor_dealii_bindings 
  ./bindings.cpp
)

PYBIND11_DEAL_II_SETUP_TARGET(pymor_dealii_bindings)
install(TARGETS pymor_dealii_bindings LIBRARY DESTINATION .)
