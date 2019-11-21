cmake_minimum_required(VERSION 3.15)
project(nonlocal_assembly)

set(CMAKE_CXX_STANDARD 14)
# add_subdirectory(src) # If more CMakeText Files are used

# Set directory of source files
set(MY_LIBRARY_SOURCE_DIR /home/klar/Promotion/projects/CLION-nonlocal-assembly-cpp/)
# Equivalent to -fIC option in the compiler
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Find OpneMP Library
find_package(OpenMP)
# Add Compiler Options
add_compile_options("-O3")
add_compile_options("-fopenmp")
add_compile_options("-Wall")

# Includes directorys in which to search when calling #include <Cassemble.h> in the source code
include_directories(${MY_LIBRARY_SOURCE_DIR}/include/)

# Some variable definitions
set(src ${MY_LIBRARY_SOURCE_DIR}/src)
set(header ${MY_LIBRARY_SOURCE_DIR}/include)

# Adds the library Cassemble. Leads to: libCassemble.a
add_library(Cassemble SHARED ${src}/Cassemble.cpp ${header}/Cassemble.h)
# Add test executable which only imports the function model_f()
add_executable(CassembleTest ${src}/test.cpp)

# Add target link libraries
target_link_libraries(CassembleTest PRIVATE Cassemble)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Cassemble PRIVATE OpenMP::OpenMP_CXX)
endif()

# When calling "Install" libCassemble.a is written to ~/lib
install(TARGETS Cassemble DESTINATION $ENV{HOME}/lib)
install(TARGETS Cassemble DESTINATION ${MY_LIBRARY_SOURCE_DIR}/lib)
