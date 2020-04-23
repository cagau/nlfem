# For some reason using OpenMP will corrupt the program! That is why I cannot simply link Cassemble libaray,
# but have to recompile the source code without OMP.
add_executable(Convergence2D ./convergence2D.cpp
        ../../src/MeshBuilder.cpp ../../include/MeshBuilder.h
        ../../src/Cassemble2D.cpp ../../include/Cassemble2D.h)
target_compile_options(Convergence2D PUBLIC "-Wall")
target_link_libraries(Convergence2D PRIVATE lapack blas armadillo)