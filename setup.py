from Cython.Distutils import build_ext
from setuptools import setup
from setuptools.extension import Extension
import os

# Environment variables
home = os.getenv("HOME")
os.environ['CC'] = 'clang'
os.environ['CXX'] = 'clang++'

# This file translates the cython code, compiles all C++ files and the translation and statically links them.
# This is convenient if you do not really plan to work in the C++ code (profiling, debugging)
# anyways and it allows building an extensive python package.
# See also https://cython.readthedocs.io/en/latest/src/tutorial/clibraries.html

# Project Name
name = "nlcfem"

ext_modules = [
    Extension(
        name=name,
        sources=["cython/nlcfem.pyx", "src/Cassemble.cpp", "src/Cassemble2D.cpp"],
        extra_link_args=['-fopenmp', '-llapack', '-lblas', '-larmadillo'],
        extra_compile_args=['-fopenmp'],
        language="c++",
        include_dirs=["include"]
    )
]

setup(
    name=name, ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    options={"bdist_wheel": {"universal": "1"}}
)
