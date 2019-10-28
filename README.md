# Usage 

In order to use this code perform the following steps.
If you use an Ubuntu machine the given shell snippets might help you.
### Basic requirements
- Download or clone this directory
- Check that you have a Python 3 with numpy, scipy, matplotlib, and Cython.
    - Cython requires a C and C++ compiler
    - <code> pip3 install Cython numpy scipy matplotlib </code>
### Compile and run
- This step translates the file assemble.pyx to a file assemble.cpp and compiles the C++ code to machine code.
    - Build: You need to setup the C++ translation of assemble.pyx. This code will not do anything if the file assemble.pyx or assemble.cpp have not changed. 
    - <code> python3 setup.py build </code>
    - Install: It should also be possible to build and install into the current directory with some command. I just dont know it.
    - <code> python3 setup.py install </code>
- Check your installation by running solve_nonlocal
    - <code> python3 solve_nonlocal.py </code>
    
### Change Model
The model is defined in the file Cassemble.cpp
- The Kernel function is called <code>  model_kernel() </code>.
- The right side is called <code>  model_f() </code>.
- The function where the assembly happens is called <code> par_assemble </code>