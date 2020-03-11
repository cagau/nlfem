# Usage 

In order to use this code perform the following steps.
### Basic requirements
- Download or clone the C++ branch of this project.
- Check that you have a Python 3 with numpy, scipy, matplotlib, and Cython available.
 (Cython requires a C and C++ compiler)
    - <code> pip3 install Cython </code>
- Check that you have CMake available

### Build and Install C++ Library Cassemble
- Enter the project directory and type
    - <code> mkdir build </code>
    - <code> cd build </code>
    - <code> cmake .. </code>
    - <code> cmake --build . --target Cassemble -- -j 4 </code>
    - <code> cmake --build . --target install -- -j 4 </code>

### Build and Install Python package assemble
- This step translates assemble.pyx to assemble.cpp (Cython) and compiles the C++ code to machine code.
    - <code> python3 setup.py build --force install</code>
    - Note, if the installation requires sudo you might want to setup a virtual environment.
- Check your installation by running solve_nonlocal
    - <code> python3 solve_nonlocal.py </code>
    
### Change Model
The model is defined in the file Cassemble.cpp
- The Kernel function is called <code>  model_kernel() </code>.
- The right side is called <code>  model_f() </code>.
- The function where the assembly happens is called <code> par_assemble </code>