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

from Cassemble cimport par_assemble, par_evaluateA, par_assemblef

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
    py_Ad = np.zeros(K_Omega* K).flatten("C")
    py_fd = np.zeros(K_Omega).flatten("C")

    cdef:
        int nP = py_P.shape[0] # Does not differ in inner and outer integral!
        long [:] c_Triangles = (np.array(Triangles, int)).flatten("C")
        double [:] c_Verts = (np.array(Verts, float)).flatten("C")
        double[:] P = (np.array(py_P, float)).flatten("C")

        # Cython interface of C-aligned arrays of solution and right side
        double[:] fd = py_fd
        double[:] Ad = py_Ad

        # List of neighbours of each triangle, Neighbours[Tdx] returns row with Neighbour indices
        long[:] Neighbours = J*np.ones((J*4), dtype=int)

        # Squared interaction horizon
        double sqdelta = pow(delta,2)

    # Setup adjaciency graph of the mesh --------------------------
    neigs = []
    for aTdx in range(J):
        neigs = get_neighbour(J, &c_Triangles[0], &c_Triangles[3*aTdx])
        n = len(neigs)
        for i in range(n):
            Neighbours[4*aTdx + i] = neigs[i]


    start = time.time()
    # Compute Assembly
    par_assemble( &Ad[0], K, &fd[0], &c_Triangles[0], &c_Verts[0], J , J_Omega, L, L_Omega, nP, &P[0], &dx[0], &dy[0], sqdelta, &Neighbours[0])
    total_time = time.time() - start
    print("Assembly\nTime needed", "{:1.2e}".format(total_time), " Sec")

    py_Ad *= 2
    return np.reshape(py_Ad, (K_Omega, K)), py_fd

cdef class clsEvaluate:
    cdef:
        int K, K_Omega, J, J_Omega, L, L_Omega, nP
        double sqdelta
        double [:] P
        double [:] dx
        double [:] dy
        long [:] c_Triangles
        double [:] c_Verts
        long [:] Neighbours

    def __init__(self,
                 # Mesh information ------------------------------------
                 int K, int K_Omega,# Number of Basis functions
                 int J, int J_Omega,# Number of Triangles and number of Triangles in Omega
                 int L, int L_Omega, # Number of vertices (in case of CG = K and K_Omega)
                 # Map Triangle index (Tdx) -> index of Vertices in Verts (Vdx = Triangle[Tdx] array of int, shape (3,))
                 long [:,:] Triangles,
                  # Map Vertex Index (Vdx) -> Coordinate of some Vertex i of some Triangle (E[i])
                 double [:,:] Verts,
                 double [:,:] py_P, # Quadrature Points
                 double [:] dx,
                 double [:] dy,
                 double delta
                 ):
        self.K = K
        self.K_Omega = K_Omega
        self.J = J
        self.J_Omega = J_Omega
        self.L = L
        self.L_Omega = L_Omega
        self.sqdelta = pow(delta,2) # Squared interaction horizon

        self.c_Triangles = (np.array(Triangles, int)).flatten("C")
        self.c_Verts = (np.array(Verts, float)).flatten("C")

        self.P = (np.array(py_P, float)).flatten("C")
        self.nP = py_P.shape[0] # Does not differ in inner and outer integral!
        self.dx = dx  # Weights for quadrature rule
        self.dy = dy

        # List of neighbours of each triangle, Neighbours[Tdx] returns row with Neighbour indices
        self.Neighbours = self.J*np.ones((self.J*4), dtype=int)

        # Setup adjaciency graph of the mesh --------------------------
        neigs = []
        for aTdx in range(self.J):
            neigs = get_neighbour(self.J, &self.c_Triangles[0], &self.c_Triangles[3*aTdx])
            n = len(neigs)
            for i in range(n):
                self.Neighbours[4*aTdx + i] = neigs[i]

    def __call__(self,
            # Input vector
            double [:] ud # Length K_Omega!
        ):

        ## Data Matrix ----------------------------------------
        # Evaluates A_Omega only, because the input is truncated
        py_vd = np.zeros(self.K_Omega).flatten("C")

        cdef:
            int aTdx=0, i=0
            # Cython interface of C-aligned arrays of solution and right side
            double[:] vd = py_vd

        start = time.time()
        # Compute Assembly
        par_evaluateA(&ud[0], &vd[0], self.K, &self.c_Triangles[0], &self.c_Verts[0], self.J , self.J_Omega,
                     self.L, self.L_Omega, self.nP, &self.P[0], &self.dx[0], &self.dy[0], self.sqdelta, &self.Neighbours[0])
        total_time = time.time() - start
        py_vd *= 2
        return py_vd

    def get_f(self):
        py_fd = np.zeros(self.K_Omega).flatten("C")
        cdef double[:] fd = py_fd
        par_assemblef(&fd[0], &self.c_Triangles[0], &self.c_Verts[0], self.J_Omega, self.L_Omega, self.nP, &self.P[0], &self.dx[0])
        return py_fd

# Setup adjaciency graph of the mesh --------------------------
cdef list get_neighbour(int rows, long * Triangles, long * Vdx):
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



