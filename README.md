# Usage 

In order to use this code perform the following steps.
### Basic requirements
- Download or clone the C++ branch of this project.
- Check that you have a Python 3 with numpy, scipy, matplotlib, and Cython available.
 (Cython requires a C and C++ compiler)
    - <code> pip3 install Cython </code>
### Build and Install
- This step translates assemble.pyx to assemble.cpp (Cython) and compiles the C++ code to machine code.
    - Note, 
    the following command will not do anything if the file assemble.pyx has not changed. 
    - <code> python3 setup.py build --force install</code>
    - Note, if the installation requires sudo you might want to setup a virtual environment.
- Check your installation by running solve_nonlocal
    - <code> python3 solve_nonlocal.py </code>
    
### Change Model
The model is defined in the file Cassemble.cpp
- The Kernel function is called <code>  model_kernel() </code>.
- The right side is called <code>  model_f() </code>.
- The function where the assembly happens is called <code> par_assemble </code>
- Note again that a change in the file Cassemble.cpp has no effect on assemble.pyx, so the build command will not 
react on those changes. You have to do some changes (possibly without effect) in assemble.pyx.