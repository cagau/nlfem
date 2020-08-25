from Cython.Distutils import build_ext
from setuptools import setup
from setuptools.extension import Extension
import os

# Environment variables
home = os.getenv("HOME")
os.environ['CC'] = 'clang'
os.environ['CXX'] = 'clang++'

# If you want to build the C++ Code into a shared library and dynamically link it to python you can
# use the CMakeLists.txt to build the binaries for the C++ code and then
# setup_shareLibrary.py. It then translates and compiles the cython code only.

# Project Name
name = "nlcfem"

ext_modules = [
    Extension(
        name=name,
        sources=["cython/nlcfem.pyx"],
        extra_link_args=['-fopenmp', '-llapack', '-lblas', '-larmadillo'],
        extra_compile_args=['-fopenmp'],
        language="c++",
        include_dirs=["include"],
        library_dirs=[home+"/lib"],
        libraries=["Cassemble"]
    )
]

# include_dirs provides the path to the header (Cassemble.h) and source (Cassemble.cpp) files
# this is necessary if we want to "inline" our code into the Cython Code

# The above case only needs the libCassemble.a or libCassemble.so and the corresponding header. See also
# https://cython.readthedocs.io/en/latest/src/tutorial/clibraries.html

setup(
    name=name, ext_modules=ext_modules, cmdclass={"build_ext": build_ext}
)

