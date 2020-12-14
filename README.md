# Collaboration Workflow

If you want to contribute follow the instructions below.

 - Create an Issue #n with a description of what should be done in the project.
 - `git checkout -b issue_n` Create a new branch on your local repository.
 - Change the code...
 - `git add x; git commit -m "Issue #n"` Commit your changes locally.
 - `git push --set-upstream origin issue_n` Push your branch to origin.
 - Create a merge request to issue_n -> master.

# Build and Install

In order to use this code you have to meet the following rewuirements
## Requirements

- The **basic requirements** for are the programs `gcc`, `g++, python3-dev, python3-venv`. You further need
the libraries `libarmadillo-dev`, and the Python 3 packages `numpy`, `Cython` and `scipy`.
- For **running the examples** you additionally need to install the Python 3 packages 
`matplotlib, pathos, meshio, meshzoo, quadpy`.

## Build and Install

You can clone the project and build and install the package via

`python3 setup.py build --force install`

The `--force` option is required if you change code outside of `nlcfem.pyx`. 
It might happen that recompilation or translation is skipped because
Cython assumes that there have been no changes.

## Step by Step Guide
To prepare the basic requirements do.

`sudo apt-get install git gcc g++ libarmadillo-dev liblapack-dev python3-venv`

`mkdir nlfemvenv`

`python3 -m venv nlfemvenv/`

`source nlfemvenv/bin/activate`

`(nlfemvenv) python3 -m pip install numpy scipy Cython`

`(nlfemvenv) python3 -m pip install matplotlib pathos meshio meshzoo quadpy`

To clone the default branch (C++) do

`git clone https://gitlab.uni-trier.de/klar/nonlocal-assembly.git path/to/nlfem`

To build and install the `nlfem` package do.

`(nlfemvenv) cd path/to/nlfem`

`(nlfemvenv) python3 setup.py build --force install`

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
