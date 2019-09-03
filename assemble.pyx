#-*- coding:utf-8 -*-
#cython: language_level=3
#cython: boundscheck=False, wraparound=False, cdivision=True
# Setting this compiler directive will given a minor increase of speed.

# Cython Imports
cimport cython
cimport numpy as np
from cython.parallel import prange, threadid
# Python Imports
import numpy as np
from numpy.linalg import LinAlgError
import time
# C Imports
from libcpp.queue cimport queue
from libc.math cimport sqrt, pow, pi, cos
from libc.stdlib cimport malloc, free, abort

from linalg cimport par_assemble

def assemble(
        # Mesh information ------------------------------------
        int K, int K_Omega,
        int J, int J_Omega, # Number of Triangles and number of Triangles in Omega
        int L, int L_Omega, # Number of vertices (in case of CG = K and K_Omega)
        # Map Triangle index (Tdx) -> index of Vertices in Verts (Vdx = Triangle[Tdx] array of int, shape (3,))
        long [:,:] Triangles,
        # Map Vertex Index (Vdx) -> Coordinate of some Vertex i of some Triangle (E[i])
        double [:,:] Verts,
        # Cython interface of quadrature points
        double [:,:] py_P,
        # Weights for quadrature rule
        double [:] dx,
        double [:] dy,
        double delta
    ):

    ## Data Matrix ----------------------------------------
    # Allocate Matrix compute_A and right side compute_f
    py_Ad = np.zeros(K_Omega* K)
    py_fd = np.zeros(K_Omega)

    # Define Basis-----------------------------------------
    cdef:
        int i=0
        # Number of integration points
        int nP = py_P.shape[0] # Does not differ in inner and outer integral!
        long [:] c_Triangles = (np.array(Triangles, int)).flatten("C")
        double [:] c_Verts = (np.array(Verts, float)).flatten("C")
        double[:] P = (np.array(py_P, float)).flatten("C")

    psi0 = np.zeros(nP)
    psi1 = np.zeros(nP)
    psi2 = np.zeros(nP)

    for i in range(nP):
        psi0[i] = (1 - py_P[i, 0] - py_P[i, 1])
        psi1[i] = py_P[i, 0]
        psi2[i] = py_P[i, 1]

    cdef:
        # Triangles -------------------------------------------------
        # Loop index of Triangles
        int aTdx=0

        # Cython interface of C-aligned arrays of solution and right side
        double[:] fd = py_fd
        double[:] Ad = py_Ad

        # Cython interface of Ansatzfunctions
        double[:] psi = (np.array([psi0, psi1, psi2], float)).flatten("C")
        # List of neighbours of each triangle, Neighbours[Tdx] returns row with Neighbour indices
        long[:] Neighbours = J*np.ones((J*4), dtype=int)

        # Squared interaction horizon
        double sqdelta = pow(delta,2)

    # Setup adjaciency graph of the mesh --------------------------
    neigs = []
    for aTdx in range(J):
        neigs = set_neighbour(J, &c_Triangles[0], &c_Triangles[3*aTdx])
        n = len(neigs)
        for i in range(n):
            Neighbours[4*aTdx + i] = neigs[i]


    start = time.time()

    # Loop over triangles ----------------------------------------------------------------------------------------------
    #for aTdx in prange(J_Omega, nogil=True):
    par_assemble( &Ad[0], K, &fd[0], &c_Triangles[0], &c_Verts[0], J , J_Omega, L, L_Omega, nP, &P[0], &dx[0], &dy[0], &psi[0], sqdelta, &Neighbours[0])

    total_time = time.time() - start
    print("\nTime needed", "{:1.2e}".format(total_time), " Sec")

    py_Ad *= 2
    return np.reshape(py_Ad, (K_Omega, K)), py_fd

cdef list set_neighbour(int rows, long * Triangles, long * Vdx):
    """
    Find neighbour for index Tdx.

    :param Triangles:
    :param Vdx:
    :return:
    """
    cdef:
        int i, j, k, n
    idx = []

    for i in range(rows):
        n = 0
        for j in range(3):
            for k in range(3):
                if Triangles[3*i+j] == Vdx[k]:
                    n+=1
        if n >= 2:
            idx.append(i)

    return idx



