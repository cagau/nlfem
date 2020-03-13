add_executable(CompareCodes ./comparecodes.cpp)
target_link_libraries(CompareCodes PRIVATE Cassemble lapack blas armadillo)