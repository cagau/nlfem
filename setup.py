from setuptools import setup
from setuptools.extension import Extension
import numpy

try:
    from Cython.Build import cythonize
    cythonize("cython/nlfem.pyx", include_path=["include"])
except ModuleNotFoundError:
    print("\nWARNING: Cython was not found. Install Cython if you want that changes in cython/nlfem.pyx have any effect!\n")

# Project Name
name = "nlfem"

ext_modules = [
    Extension(
        name=name,
        sources=["cython/nlfem.cpp",
                 "src/Cassemble.cpp",
                 "./src/MeshTypes.cpp",
                 "./src/mathhelpers.cpp",
                 "./src/model.cpp",
                 "./src/integration.cpp"],
        extra_link_args=['-larmadillo', '-lCGAL', '-lgmp', '-lmpfr', '-fopenmp'],
        extra_compile_args=['-O3', '-DARMA_NO_DEBUG', '-fopenmp'],
        language="c++",
        include_dirs=["include", "src", numpy.get_include()]
    )
]

setup(
    name=name,
    ext_modules=ext_modules,
    version="0.0.1",
    author="Christian Vollmann, Manuel Klar",
    author_email="vollmann@uni-trier.de, klar@uni-trier.de",
    description="This library provides a parallel assembly routine for a specific class of integral operators.",
    python_requires='>=3.6',
    include_package_data=True,
    install_requires=[['numpy'], ['scipy'], ['Cython']]
)
