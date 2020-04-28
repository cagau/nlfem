from Cython.Distutils import build_ext
from distutils.core import setup
import setuptools
from distutils.extension import Extension
import os

# Environment variables
home = os.getenv("HOME")
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

# Project Name
name = "assemble"
pdir=""
sources=[ "src/Cassemble.cpp", "src/Cassemble2D.cpp", "cython/assemble.pyx", "include/Cassemble.pxd",
          "include/Cassemble2D.pxd"]
include_dirs=["include", "cython", "src"]
sources=[pdir+path for path in sources]
include_dirs=[pdir+path for path in include_dirs]

ext_modules = [
    Extension(
        name=name,
        sources=sources,
        extra_link_args=['-llapack', '-lblas', '-lgomp', '-larmadillo'],
        language="c++",
        include_dirs=include_dirs,
        extra_compile_args=['-fopenmp', '-O2']
    )
]

# include_dirs provides the path to the header (Cassemble.h) and source (Cassemble.cpp) files
# this is necessary if we want to "inline" ou code into the Cython Code

# The above case only needs the libCassemble.a or libCassemble.so and the corresponding header. See also
# https://cython.readthedocs.io/en/latest/src/tutorial/clibraries.html

setup(
    name=name,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext}
)
