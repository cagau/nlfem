# Build and Install

In order to use this code you have to meet the following rewuirements
## Requirements

The **basic requirements** for python are `numpy` and `scipy`.
For **running the examples** you additionally need to install `matplotlib, pathos, meshio, meshzoo, quadpy`.
If you want to **build the binaries** on your system you also need the
python package `Cython` as well as `gcc`, and `cmake`.

## Install with pip

The package `nlcfem` can be installed using `pip` via

`pip install nlcfem --index-url https://__token__:<read-api-token>@gitlab.uni-trier.de/api/v4/projects/402/packages/pypi/simple --no-deps`

You need a personal access toke to allow pip to access gitlab. Find more 
information under https://gitlab.uni-trier.de/help/api/README.md#personalproject-access-tokens.

## Build as static library

You can clone the project and build and install the package via

`python3 setup.py build_ext --force install`

The `--force` option is required if you change code outside of `nlcfem.pyx`. 
It might happen that recompilation or translation is skipped because
Cython assumes that there have been no changes.

## Build as shared library

### Build and Install the C++ Library Cassemble.so
If you want to install the project as standalone shared object file you have to
- Enter the project directory and type
    - <code> mkdir build </code>
    - <code> cd build </code>
    - <code> cmake .. </code>
    - <code> cmake --build . --target Cassemble -- -j 4 </code>
    - <code> cmake --build . --target install -- -j 4 </code>
- `libCassemble.so` will be installed to <code> /home/user/lib </code> add this path via
    - <code> export LD_LIBRARY_PATH="/home/user/lib" </code>
The shared object file can then be dynamically linked to the nlcfem package.

### Build and Install nlcfem while dynamically linking Cassemble.so
- This step translates assemble.pyx to assemble.cpp (Cython) and compiles the C++ code to machine code.
    - <code> python3 setup_sharedLibrary.py build --force install</code>
    - Note, if the installation requires sudo you might want to setup a virtual environment.
- Check your installation by running solve_nonlocal
    - <code> python3 solve_nonlocal.py </code>
    
- If cmake cannot be found try installing it. If it still cannot be found try reloading the CMake Project.

# Usage

Test the code using the examples.

## Changing the Operator or the right hand side
If you want to change the model you have to alter the file `src/model.cpp`. You find
a several options for kernel and forcing functions there. 

- The Kernel function is called <code>  model_kernel() </code>.
- The right side is called <code>  model_f() </code>.

### Step 1 Altering the model
If you want to change the kernel implement your version
`kernel_myKernel()` while exactly meeting the interface of the function pointer
`mode_kernel`. Do the same thing for the right hand side.

### Step 2 Adding the option
Open the file `src/Cassemble.cpp` and add an if clause in the function
`lookup_configuration()` of the type `conf.model_kernel == "myKernel"` in order
to make your kernel available from the outside.
The function where the assembly happens is called <code> par_system </code> and
is also to be found in this file.

# Documentation

Here you find the 
[documentation](http://klar.gitlab-pages.uni-trier.de/nonlocal-assembly/)
of the C++ Code.
