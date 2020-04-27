# cmake version to be used
cmake_minimum_required(VERSION 3.10)

# project name
project(nonlocal_assembly)

# flags ----------------------------------------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
add_compile_options("-O2")
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # Equivalent to -fPIC option in the compiler

# files ----------------------------------------------------------------------------------------------------------------
# Abbreviations for file names...

# include --------------------------------------------------------------------------------------------------------------
include_directories(./include/ ./src/)

# target ---------------------------------------------------------------------------------------------------------------
# Adds the library Cassemble. Leads to: libCassemble.a
add_library(Cassemble SHARED ./src/Cassemble.cpp ./include/Cassemble.h
        ./src/MeshBuilder.cpp ./include/MeshBuilder.h
        ./src/Cassemble2D.cpp ./include/Cassemble2D.h)
target_compile_options(Cassemble PUBLIC "-fopenmp" "-Wall")

# external libraries ---------------------------------------------------------------------------------------------------
target_link_libraries(Cassemble PRIVATE lapack blas armadillo)
# Find OpneMP Library
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Cassemble PRIVATE OpenMP::OpenMP_CXX)
endif()

add_subdirectory(./examples/Rates2DCpp)
add_subdirectory(./examples/CompareCodes)
# Add tests to project
#option(PACKAGE_TESTS "Build the tests" ON)
#if(PACKAGE_TESTS)
#    enable_testing()
#    add_subdirectory(tests)
#endif()

# install --------------------------------------------------------------------------------------------------------------
# When calling "Install" libCassemble.a is written to ~/lib
install(TARGETS Cassemble DESTINATION $ENV{HOME}/lib)
