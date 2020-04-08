# In this file all declarations of the C functions happen, in order to make them visible to cimport.
# If this .pxd is compiled with setup_linalg.py it results in a standalone package which can be cimported.
# In order import (to python) see linalg.pyx

# distutils: language = c++
cimport armadillo

cdef extern from "Cassemble.h":
    # Assembly algorithm with BFS ----

