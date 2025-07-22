cmake_minimum_required(VERSION 3.13.4)
project(deal_ii_testbench VERSION 0.0.1 LANGUAGES CXX)

# Set the C++ standard to C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build (Debug or Release)" FORCE)
endif()

# Set custom output directory for binaries
string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type_lc)
set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin/${build_type_lc}")

# Print selected build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Enabling debug flags (-g -O0)")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0" CACHE STRING "Debug flags" FORCE)
endif()

#set(DDEAL_II_WITH_SUPERLU ON)

find_package(deal.II 9.6.0 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  COMPONENTS PETSC SLEPC
  )

deal_ii_initialize_cached_variables()

find_package(pybind11 CONFIG REQUIRED)
INCLUDE(dealii_macro.cmake)
DEAL_II_INITIALIZE_CACHED_VARIABLES()

add_subdirectory(material_model)
add_subdirectory(pymor_dealii_bindings)