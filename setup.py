from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension

import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

ext_modules = [
    Extension(
        name="assemble",
        sources=["assemble.pyx", "Cassemble.pxd"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function", "-O2", "-march=native",  '-std=c++11', '-fopenmp'],
        extra_link_args=["-O2", "-march=native", '-fopenmp'],
        language="c++",
        include_dirs=["."],
    )
]

setup(
    name="assemble", ext_modules=ext_modules, cmdclass={"build_ext": build_ext}
)

