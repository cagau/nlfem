from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension

import os
os.environ['CC'] = 'g++'
os.environ['CXX'] = 'g++'

ext_modules = [
    Extension(
        name="linalg",
        sources=["linalg.pxd", "linalg.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function", "-O2", "-march=native",  '-std=c++11'],
        extra_link_args=["-O2", "-march=native"],
        language="c++",
        include_dirs=["."],
    )
]

setup(
    name="linalg", ext_modules=ext_modules, cmdclass={"build_ext": build_ext}
)

