from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(['assemble.py', 'aux.py', 'conf.py', 'nbhd.py', 'nlocal.py', 'plot.py', 'solve_nonlocal.py'])
)