# Build and Install

In order to use this code you have to meet the following requirements
## Requirements

- The **basic requirements** for are the programs 
  `gcc, g++, python3-dev, python3-venv, libgmp-dev, libcgal-dev, metis, libmetis-dev, libarmadillo-dev`. 
  You further need the Python 3 packages `numpy` and `scipy`. If you want to change
  the Cython code, you require the package `Cython`.
- For **running the examples** you additionally need to install the Python packages 
`matplotlib, meshzoo`. Make sure to install the correct versions by using the `requirements.txt` as given below.

## Build and Install

You can clone the project and build and install the package via

`python3 setup.py build --force install`

The `--force` option is required if you change code outside of `nlcfem.pyx`. 
It might happen that recompilation or translation is skipped because
Cython assumes that there have been no changes.

## Step by Step Guide
To prepare the basic requirements on Ubuntu do.

`sudo apt-get install git gcc g++ libarmadillo-dev liblapack-dev python3-venv python3-dev libgmp-dev libcgal-dev`

`mkdir nlfemvenv`

`python3 -m venv nlfemvenv/`

`source nlfemvenv/bin/activate`

`(nlfemvenv) python3 -m pip install -r requirements.txt`

To clone the default branch (master) do

`git clone https://gitlab.uni-trier.de/klar/nonlocal-assembly.git path/to/nlfem`

To build and install the `nlfem` package do.

`(nlfemvenv) cd path/to/nlfem`

`(nlfemvenv) python3 setup.py build --force install`

# Usage

To test the rates for the constant kernel run

`(nlfemvenv) cd path/to/nlfem/examples/Test2D`

`(nlfemvenv) python3 computeRates2D.py -f testConfConstant -s 4`

Run a more extensive test via the option `-f testConfFull -s 4`. Get help via the option `-h`.

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

Find a documentation of the C++ code in `docs/html`.

# Licence

nlfem is published under MIT license.

Copyright (c) 2021 Manuel Klar, Christian Vollmann