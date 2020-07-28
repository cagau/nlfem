# Usage 

In order to use this code perform the following steps.
### Basic requirements
- Download or clone the C++ branch of this project.
- Check that you have a Python 3 with numpy, scipy, matplotlib, pathos, meshio and Cython available.
 (Cython requires a C and C++ compiler)
    - <code> pip3 install Cython scipy pathos matplotlib numpy meshzoo meshio quadpy </code>
- Check that you have CMake available

    
### Change Model
The model is defined in the file Cassemble.cpp
- The Kernel function is called <code>  model_kernel() </code>.
- The right side is called <code>  model_f() </code>.
- The function where the assembly happens is called <code> par_system </code>

### Build and Install Python package assemble
- This step translates assemble.pyx to assemble.cpp (Cython) and compiles the C++ code to machine code.
    - <code> python3 setup.py build --force install</code>
    - Note, if the installation requires sudo you might want to setup a virtual environment.
- Check your installation by running solve_nonlocal
    - <code> python3 solve_nonlocal.py </code>
 
### [Documentation](http://klar.gitlab-pages.uni-trier.de/nonlocal-assembly/Cassemble_8h.html#a6c9fe18c400dcd08c34d7655499f8375)

### [Optional] Build and Install the C++ Library Cassemble.so
If you want to install the project as standalone shared object file you have to
- Enter the project directory and type
    - <code> mkdir build </code>
    - <code> cd build </code>
    - <code> cmake .. </code>
    - <code> cmake --build . --target Cassemble -- -j 4 </code>
    - <code> cmake --build . --target install -- -j 4 </code>
- libCassemble.so will be installed to <code> /home/user/lib </code> add this path via
    - <code> export LD_LIBRARY_PATH="/home/user/lib" </code>
The shared object file can then be dynamically linked to the assemble package.
### [Optional] Build and Install assemble while dynamically linking Cassemble.so
- This step translates assemble.pyx to assemble.cpp (Cython) and compiles the C++ code to machine code.
    - <code> python3 setup_sharedLibrary.py build --force install</code>
    - Note, if the installation requires sudo you might want to setup a virtual environment.
- Check your installation by running solve_nonlocal
    - <code> python3 solve_nonlocal.py </code>
