# Set directory of source files
set(MY_LIBRARY_SOURCE_DIR $ENV{HOME}/Promotion/projects/CLION-nonlocal-assembly-cpp)

# Includes directorys in which to search when calling #include <Cassemble.h> in the source code
include_directories(${MY_LIBRARY_SOURCE_DIR}/include/)

# Some variable definitions
set(src ${MY_LIBRARY_SOURCE_DIR}/src)
set(exp ${MY_LIBRARY_SOURCE_DIR}/examples/CompareCodes)
set(header ${MY_LIBRARY_SOURCE_DIR}/include)
add_executable(CompareCodes ${exp}/comparecodes.cpp ${header}/MeshBuilder.h ${src}/MeshBuilder.cpp ${header}/Cassemble2D.h ${src}/Cassemble2D.cpp ${header}/Cassemble.h ${src}/Cassemble.cpp)
target_link_libraries(CompareCodes PRIVATE lapack blas armadillo)