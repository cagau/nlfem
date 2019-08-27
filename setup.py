#from distutils.core import setup
#from Cython.Build import cythonize#
#
#setup(
#    ext_modules=cythonize(['assemble.pyx'], compiler_directives={"extra_compile_flags":"-stdlib=libc++"})
#)

from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension

import os
os.environ['CC'] = 'g++'
os.environ['CXX'] = 'g++'

ext_modules = [
    Extension(
        name="assemble",
        sources=["assemble.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function", "-O2", "-march=native",  '-std=c++11'],
        extra_link_args=["-O2", "-march=native"],
        language="c++",
        include_dirs=["."],
    )
]

setup(
    name="assemble", ext_modules=ext_modules, cmdclass={"build_ext": build_ext}
)