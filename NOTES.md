
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

## Install with pip

The package `nlcfem` can be installed using `pip` via

`pip install nlcfem --index-url https://__token__:<read-api-token>@gitlab.uni-trier.de/api/v4/projects/402/packages/pypi/simple --no-deps`

You need a personal access toke to allow pip to access gitlab. Find more 
information under https://gitlab.uni-trier.de/help/api/README.md#personalproject-access-tokens.