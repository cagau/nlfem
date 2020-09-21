from Cython.Distutils import build_ext
from setuptools import setup
from setuptools.extension import Extension
import os
import numpy

# Environment variables
home = os.getenv("HOME")
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

# This file translates the cython code, compiles all C++ files and the translation and statically links them.
# This is convenient if you do not really plan to work in the C++ code (profiling, debugging)
# anyways and it allows building an extensive python package.
# See also https://cython.readthedocs.io/en/latest/src/tutorial/clibraries.html

# Project Name
name = "nlfem"

ext_modules = [
    Extension(
        name=name,
        sources=["cython/nlfem.pyx", "src/Cassemble.cpp",
                    "./include/MeshTypes.cpp",
                    "./src/mathhelpers.cpp",
                    "./src/model.cpp",
                    "./src/integration.cpp"],
        extra_link_args=['-fopenmp', '-llapack', '-lblas', '-larmadillo'],
        extra_compile_args=['-fopenmp'],
        language="c++",
        include_dirs=["include", numpy.get_include()]
    )
]

setup(
    name=name, ext_modules=ext_modules,
    version="0.0.1",
    author="Christian Vollmann, Manuel Klar",
    author_email="vollmann@uni-trier.de, klar@uni-trier.de",
    description="This library provides a parallel assembly routine for a specific class of integral operators. ",
    python_requires='>=3.6',
    include_package_data=True,
    install_requires=[['numpy'], ['scipy'], ['Cython']],
    cmdclass={"build_ext": build_ext},
    options={"bdist_wheel": {"universal": "1"}}
)