add_executable(CompareCodes ./comparecodes.cpp)
target_compile_options(Cassemble PUBLIC "-fopenmp" "-Wall")
target_link_libraries(CompareCodes PRIVATE Cassemble lapack blas armadillo OpenMP::OpenMP_CXX metis)
install(TARGETS CompareCodes DESTINATION ${PROJECT_SOURCE_DIR}/examples/CompareCodes/)