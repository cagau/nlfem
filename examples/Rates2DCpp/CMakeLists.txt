#cmake_minimum_required(VERSION 3.10)
project(nonlocal_assembly)

#set(CMAKE_CXX_STANDARD 14)
# Set directory of source files
set(MY_LIBRARY_SOURCE_DIR $ENV{HOME}/Promotion/projects/CLION-nonlocal-assembly-cpp)
# Equivalent to -fPIC option in the compiler
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Find OpneMP Library
#find_package(OpenMP)
# Add Compiler Options
#add_compile_options("-O3")
#add_compile_options("-fopenmp")
#add_compile_options("-Wall")

# Includes directorys in which to search when calling #include <Cassemble.h> in the source code
include_directories(${MY_LIBRARY_SOURCE_DIR}/include/)

# Some variable definitions
set(src ${MY_LIBRARY_SOURCE_DIR}/src)
set(exp ${MY_LIBRARY_SOURCE_DIR}/examples/Rates2DCpp)
message( ${exp})
set(header ${MY_LIBRARY_SOURCE_DIR}/include)

# Adds the library Cassemble. Leads to: libCassemble.a
add_executable(Convergence2D ${exp}/convergence2D.cpp ${header}/MeshBuilder.h ${src}/MeshBuilder.cpp ${header}/Cassemble2D.h ${src}/Cassemble2D.cpp)
#if(OpenMP_CXX_FOUND)
#    target_link_libraries(Convergence2D PRIVATE OpenMP::OpenMP_CXX)
#endif()
target_link_libraries(Convergence2D PRIVATE lapack blas armadillo)