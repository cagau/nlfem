from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension

import os
import sys
print(sys.path)
os.environ['CC'] = 'clang'
os.environ['CXX'] = 'clang++'
# Project Name
name="assemble"

ext_modules = [
    Extension(
        name=name,
        sources=["assemble.pyx"],
        # -fopenmp not necessary in clang
        # optimization happens in compilation of main libraray
        # extra_compile_args=["-Wall", "-O3", "-march=native", '-fopenmp'],
        extra_link_args=['-fopenmp'],
        language="c++",
        include_dirs=["../include"],#, "../src"],
        library_dirs=["../lib"],
        libraries=["Cassemble"]
    )
]

# include_dirs provides the path to the header (Cassemble.h) and source (Cassemble.cpp) files
# recompiling the library is the way how to statically link the
# Cpp-Code into the Cython object. See also
# https://cython.readthedocs.io/en/latest/src/tutorial/clibraries.html

setup(
    name=name, ext_modules=ext_modules, cmdclass={"build_ext": build_ext}
)

