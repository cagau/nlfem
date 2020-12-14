from Cython.Distutils import build_ext
from setuptools import setup
from setuptools.extension import Extension
import numpy

# Project Name
name = "nlfem"

ext_modules = [
    Extension(
        name=name,
        sources=["cython/nlfem.pyx",
                    "src/Cassemble.cpp",
                    "./src/MeshTypes.cpp",
                    "./src/mathhelpers.cpp",
                    "./src/model.cpp",
                    "./src/integration.cpp"],
        extra_link_args=['-fopenmp', '-larmadillo'],
        extra_compile_args=['-fopenmp'],
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
    description="This library provides a parallel assembly routine for a specific class of integral operators. ",
    python_requires='>=3.6',
    include_package_data=True,
    install_requires=[['numpy'], ['scipy'], ['Cython']],
    cmdclass={"build_ext": build_ext}
    #options={"bdist_wheel": {"universal": "1"}}
)
