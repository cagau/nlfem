cmake_minimum_required(VERSION 3.15)
project(nonlocal_assembly)

set(CMAKE_CXX_STANDARD 14)
# add_subdirectory(src) # If more CMakeText Files are used

# Set directory of source files
set(MY_LIBRARY_SOURCE_DIR $ENV{HOME}/Promotion/projects/CLION-nonlocal-assembly-cpp/)
# Equivalent to -fPIC option in the compiler
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Find OpneMP Library
find_package(OpenMP)
# Add Compiler Options
add_compile_options("-O0")
add_compile_options("-fopenmp")
add_compile_options("-Wall")

# Includes directorys in which to search when calling #include <Cassemble.h> in the source code
include_directories(${MY_LIBRARY_SOURCE_DIR}/include/)

# Add tests to project
option(PACKAGE_TESTS "Build the tests" ON)
if(PACKAGE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Some variable definitions
set(src ${MY_LIBRARY_SOURCE_DIR}/src)
set(header ${MY_LIBRARY_SOURCE_DIR}/include)

# Adds the library Cassemble. Leads to: libCassemble.a
add_library(Cassemble SHARED ${src}/Cassemble.cpp ${header}/Cassemble.h src/MeshBuilder.cpp src/MeshBuilder.h src/MeshBuilder.cpp src/MeshBuilder.cpp src/MeshBuilder.h)
# Add test executable which only imports the function model_f()
add_executable(CassembleTest ${src}/MeshBuilder.cpp)

# Add target link libraries
#target_link_libraries(CassembleTest PRIVATE)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Cassemble PRIVATE OpenMP::OpenMP_CXX)
endif()
target_link_libraries(Cassemble PRIVATE lapack blas armadillo)

# When calling "Install" libCassemble.a is written to ~/lib
install(TARGETS Cassemble DESTINATION $ENV{HOME}/lib)