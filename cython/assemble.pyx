#-*- coding:utf-8 -*-
# distutils: include_dirs = ../include

#cython: language_level=3
#cython: boundscheck=True, wraparound=True, cdivision=False
# Setting this compiler directive will given a minor increase of speed.

# Assembly routine
cimport Cassemble
#from Cassemble cimport par_assemble
import numpy as np
import time
from libc.math cimport pow
from cpython.mem cimport PyMem_Malloc, PyMem_Free

def assemble(
        # Mesh information ------------------------------------
        mesh,
        double [:,:] py_Px,
        double [:,:] py_Py,
        # Weights for quadrature rule
        double [:] dx,
        double [:] dy,
        double delta,
        **kwargs
    ):

    cdef:
        int is_DiscontinuousGalerkin
        int is_NeumannBoundary
        long nE = mesh.nE
        long nV = mesh.nV
        long nE_Omega = mesh.nE_Omega
        long nV_Omega = mesh.nV_Omega

    if mesh.boundaryConditionType == "Dirichlet":
        is_NeumannBoundary = 0
    else:# mesh.boundaryConditionType == "Neumann": #
        is_NeumannBoundary = 1
        nE_Omega = nE
        nV_Omega = nV

    if mesh.ansatz=="DG":
        K = mesh.nE*3
        K_Omega = nE_Omega*3
        is_DiscontinuousGalerkin=1
    elif mesh.ansatz=="CG":
        K = mesh.nV
        K_Omega = nV_Omega
        is_DiscontinuousGalerkin=0
    else:
        print("Ansatz ", mesh.ansatz, " not provided.")
        raise ValueError



    deltaVertices = kwargs.get("deltaVertices", None)
    if not deltaVertices is None:
        Verts = mesh.vertices + deltaVertices
    else:
        Verts = mesh.vertices

    ## Data Matrix ----------------------------------------
    # Allocate Matrix compute_A and right side compute_f
    py_Ad = np.zeros(K_Omega* K).flatten("C")
    py_fd = np.zeros(K_Omega).flatten("C")


    cdef:
        int aTdx, n , i
        int nPx = py_Px.shape[0]
        int nPy = py_Py.shape[0]
        long [:] c_Triangles = (np.array(mesh.triangles, int)).flatten("C")
        double [:] c_Verts = (np.array(Verts, float)).flatten("C")
        double[:] Px = (np.array(py_Px, float)).flatten("C")
        double[:] Py = (np.array(py_Py, float)).flatten("C")
        # Cython interface of C-aligned arrays of solution and right side
        double[:] fd = py_fd
        double[:] Ad = py_Ad

        # List of neighbours of each triangle, Neighbours[Tdx] returns row with Neighbour indices
        long[:] Neighbours = mesh.nE*np.ones((mesh.nE*4), dtype=int)

        # Squared interaction horizon
        double sqdelta = pow(delta,2)

    # Setup adjaciency graph of the mesh --------------------------
    neigs = []
    for aTdx in range(nE):
        neigs = get_neighbour(nE, &c_Triangles[0], &c_Triangles[4*aTdx])
        n = len(neigs)
        for i in range(n):
            Neighbours[4*aTdx + i] = neigs[i]


    start = time.time()
    # Compute Assembly
    Cassemble.par_assemble( &Ad[0], K_Omega, K, &fd[0], &c_Triangles[0], &c_Verts[0], nE , nE_Omega, nV, nV_Omega, &Px[0], nPx, &dx[0], &Py[0], nPy, &dy[0], sqdelta, &Neighbours[0], is_DiscontinuousGalerkin, is_NeumannBoundary)
    total_time = time.time() - start
    print("Assembly Time\t", "{:1.2e}".format(total_time), " Sec")

    py_Ad *= 2
    return np.reshape(py_Ad, (K_Omega, K)), py_fd

# Setup adjaciency graph of the mesh --------------------------
cdef list get_neighbour(int rows, long * Triangles, long * Vdx):
    cdef:
        int i, j, k, n
    idx = []

    for i in range(rows):
        n = 0
        for j in range(3):
            for k in range(3):
                if Triangles[4*i+1+j] == Vdx[k+1]:
                    n+=1
        if n == 2:
            idx.append(i)

    return idx

# DEBUG Helpers - -----------------------------------------------------------------------------------------------------
from Cassemble cimport retriangulate
from Cassemble cimport toPhys, toRef, model_basisFunction
import matplotlib
from libcpp.queue cimport queue
import matplotlib.pyplot as plt

def py_retriangulate(
        double [:] x_center,
        double [:] TE,
        double delta,
        int is_placePointOnCaps,
        pp
    ):

    TriangleList =  np.zeros(9*3*2)
    cdef:
        double sqdelta = pow(delta,2)
        double [:] cTriangleList = TriangleList
    Rdx = retriangulate(&x_center[0], &TE[0], sqdelta, &cTriangleList[0], is_placePointOnCaps)

    return Rdx, TriangleList

def py_check_par_assemble(
        # Mesh information ------------------------------------
        mesh,
        double [:,:] py_P,
        # Weights for quadrature rule
        double [:] dx,
        double delta,
        **kwargs
    ):

    pp = kwargs.get("pp", None)

    K = mesh.nE

    py_fd = np.zeros(K).flatten("C")

    cdef:
        long nE = mesh.nE
        long nV = mesh.nV
        long nE_Omega = mesh.nE_Omega
        long nV_Omega = mesh.nV_Omega

        int nP = py_P.shape[0] # Does not differ in inner and outer integral!
        long [:] Triangles = (np.array(mesh.triangles, int)).flatten("C")
        double [:] Verts = (np.array(mesh.vertices, float)).flatten("C")
        double[:] P = (np.array(py_P, float)).flatten("C")

        # Cython interface of C-aligned arrays of solution and right side
        double[:] fd = py_fd
        # List of neighbours of each triangle, Neighbours[Tdx] returns row with Neighbour indices
        long[:] Neighbours = mesh.nE*np.ones((mesh.nE*4), dtype=int)
        # Squared interaction horizon
        double sqdelta = pow(delta,2)

    # Setup adjaciency graph of the mesh --------------------------
    neigs = []
    for aTdx in range(mesh.nE):
        neigs = get_neighbour(mesh.nE, &Triangles[0], &Triangles[4*aTdx])
        n = len(neigs)
        for i in range(n):
            Neighbours[4*aTdx + i] = neigs[i]

    start = time.time()

    cdef int  h=0
    ## General Loop Indices ---------------------------------------
    cdef int bTdx=0

    ## Breadth First Search --------------------------------------
    cdef visited = np.zeros(mesh.nE)#(int *) malloc(J*sizeof(int));

    ## Loop index of current outer triangle in BFS
    cdef int sTdx=0
    ## Queue for Breadth first search
    cdef queue[int] queue
    ## List of visited triangles
    cdef long *NTdx
    ## Determinant of Triangle a and b.
    cdef  double aTdet, bTdet, termarea
    ## Vector containing the coordinates of the vertices of a Triangle
    cdef double aTE[2*3]
    cdef double bTE[2*3]
    ## Integration information ------------------------------------
    ##// Loop index of basis functions
    cdef int a=0, b=0, aAdxj =0
    cdef long labela=0, labelb=0
    #cdef double [:] Verts = mesh.vertices.flatten("C")
    #cdef long [:] Triangles = mesh.triangles.flatten("C")

    for aTdx in [7,16, 41]:#, 95, 98, 100]:#(mesh.nE-10, mesh.nE):
        fig = plt.figure()  # create a figure object
        ax = fig.add_subplot(1, 1, 1)
        ## Prepare Triangle information aTE and aTdet ------------------
        ## Copy coordinates of Triange a to aTE.
        ## this is done fore convenience only, actually those are unnecessary copies!
        print(aTdx)
        for j in range(2):
            aTE[2*0+j] = Verts[2*Triangles[4*aTdx+1] + j]
            aTE[2*1+j] = Verts[2*Triangles[4*aTdx+2] + j]
            aTE[2*2+j] = Verts[2*Triangles[4*aTdx+3] + j]

        ## compute Determinant
        aTdet = absDet(aTE)
        labela = Triangles[4*aTdx]

        ## Of course some uneccessary computation happens but only for some verticies of thos triangles which lie
        ## on the boundary. This saves us from the pain to carry the information (a) into the integrator compute_f.

        ## BFS -------------------------------------------------------------
        ## Intialize search queue with current outer triangle
        queue.push(aTdx)
        ## Initialize vector of visited triangles with 0
        visited.fill(0)
        h=0
        ## Check whether BFS is over.
        while ( not queue.empty()):
            h+=1
            ## Get and delete the next Triangle index of the queue. The first one will be the triangle aTdx itself.
            sTdx = queue.front()
            queue.pop()
            ## Get all the neighbours of sTdx.
            NTdx =  &Neighbours[4*sTdx]
            ## Run through the list of neighbours. (4 at max)
            for j in range(4):
                ## The next valid neighbour is our candidate for the inner Triangle b.
                bTdx = NTdx[j]

                ## Check how many neighbours sTdx has. It can be 4 at max. (Itself, and the three others)
                ## In order to be able to store the list as contiguous array we fill up the empty spots with the number J
                ## i.e. the total number of Triangles (which cannot be an index).
                if (bTdx < mesh.nE):

                    ## Prepare Triangle information bTE and bTdet ------------------
                    ## Copy coordinates of Triange b to bTE.
                    ## again this is done fore convenience only, actually those are unnecessary copies!
                    for i in range(2):
                        bTE[2*0+i] = Verts[2*Triangles[4*bTdx+1] + i]
                        bTE[2*1+i] = Verts[2*Triangles[4*bTdx+2] + i]
                        bTE[2*2+i] = Verts[2*Triangles[4*bTdx+3] + i]

                    bTdet = absDet(bTE)
                    labelb = Triangles[4*bTdx]
                    ## Check wheter bTdx is already visited.
                    if (visited[bTdx]==0):

                        ## Assembly of matrix ---------------------------------------
                        ## Compute integrals and write to buffer
                        #termarea = 1
                        termarea = compute_area(aTE, aTdet, labela, bTE, bTdet, labelb, P, nP, dx, sqdelta, pp, ax, aTdx, bTdx, h)
                        ## If bT interacts it will be a candidate for our BFS, so it is added to the queue
                        if (termarea > 0):
                            queue.push(bTdx)
                            fd[aTdx] += termarea
                    visited[bTdx] = 1
        if pp is not None:
            plt.savefig(pp, format='pdf')
            plt.close()

    total_time = time.time() - start
    print("Assembly Time\t", "{:1.2e}".format(total_time), " Sec")


    return py_fd

cdef absDet(double [:] E):
    cdef double [:,:] M = np.zeros((2,2))
    cdef int i=0
    for i in range(2):
        M[i][0] = E[2*1+i] - E[2*0+i]
        M[i][1] = E[2*2+i] - E[2*0+i]

    return np.abs(M[0][0]*M[1][1] - M[0][1]*M[1][0])

cdef double compute_area(double [:] aTE, double aTdet, long labela, double [:] bTE, double bTdet, long labelb, double [:] P, int nP, double [:] dx, double sqdelta, pp, ax, aTdx, bTdx, h):
    cdef:
        double areaTerm=0.0
        int rTdx, Rdx, s_max=15
        double * x
        double physical_quad[2], reference_quad[2], psi_value[3]
        double [:] reTriangle_list = np.zeros([9*3*2])

    x = &P[2*15]
    toPhys(&aTE[0], x, &physical_quad[0])
    py_IntegrationPoint = np.array(physical_quad)
    TE = np.array(bTE)
    TE = TE.reshape((3,2))
    if pp is not None:
        ax.fill(TE[:, 0], TE[:, 1], color="b", alpha=1, fill=False,  lw=.1)
        ax.scatter(py_IntegrationPoint[0], py_IntegrationPoint[1], c = "blue", s=2)
    is_placePointOnCaps = 1
    Rdx = retriangulate(physical_quad, &bTE[0], sqdelta, &reTriangle_list[0], is_placePointOnCaps)
    for rTdx in range(Rdx):
        areaTerm += absDet(reTriangle_list[2*3*rTdx:])/2
        TE = np.array(reTriangle_list[2*3*rTdx:(2*3*rTdx + 6)])
        TE = TE.reshape((3,2))
        if pp is not None:
            plt.title(aTdx)
            plt.gca().set_aspect('equal')
            ax.fill(TE[:, 0], TE[:, 1], color="blue", alpha=.3)
            ax.fill(TE[:, 0], TE[:, 1], color="blue", fill=False,  lw=.7, alpha=.3)
            if bTdx == aTdx:
                ax.fill(TE[:, 0], TE[:, 1], color="blue", lw=.7, alpha=.3)
                physical_quad_List = np.zeros((nP,2))
                psi_value_List = np.zeros(nP)
                for i in range(0):
                    toPhys(&reTriangle_list[2*3*rTdx], &P[2*i], &physical_quad[0])
                    py_physical_quad = np.array(physical_quad)
                    ax.scatter(py_physical_quad[0], py_physical_quad[1], c="black", s=.1, alpha=.5)
                    ax.annotate(i, py_physical_quad,alpha=.7, fontsize=2)
                    toRef(&bTE[0], &physical_quad[0], &reference_quad[0])
                    model_basisFunction(&reference_quad[0], &psi_value[0])
                    #print(np.array(psi_value))
                    #ax.scatter(py_physical_quad[0], py_physical_quad[1], c="black", s=psi_value[1]*s_max, alpha=.9)
                    #ax.tricontour(py_physical_quad[0], py_physical_quad[1], Z=psi_value[1])
                    py_bTE = np.array(bTE)
                    py_bTE = py_bTE.reshape((3,2))
                    ax.scatter(py_bTE[:,0], py_bTE[:,1], s=3, alpha=1)
                    for k in range(3):
                        ax.annotate("Psi"+str(k) ,py_bTE[k], fontsize=3, alpha=.1)
                    physical_quad_List[i] = py_physical_quad
                    psi_value_List[i] =  psi_value[0]
                #ax.tricontourf(physical_quad_List[:,0], physical_quad_List[:,1], psi_value_List,  5, cmap=plt.cm.get_cmap('rainbow'), vmin=0, vmax=1)
                #ax, _ = matplotlib.colorbar.make_axes(plt.gca(), shrink=.7)
                #matplotlib.colorbar.ColorbarBase(ax, cmap=plt.cm.get_cmap('rainbow'),
                #                             norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
    if pp is not None:
        ax.annotate(bTdx, baryCenter(bTE), fontsize=5)

    return areaTerm

cdef double [:] baryCenter(double [:] E):
    cdef:
        int i=0
        double [:] bary = np.zeros(2)
    for i in range(3):
        bary[0] += E[2*i]
        bary[1] += E[2*i+1]
    bary[0] = bary[0]/3
    bary[1] = bary[1]/3
    return bary

cdef toPhys_(double [:] E, double * p, double * out_x):
    cdef int i=0
    for i in range(2):
        out_x[i] = (E[2*1+i] - E[2*0+i])*p[0] + (E[2*2+i] - E[2*0+i])*p[1] + E[2*0+i]

# [END] DEBUG Helpers - ------------------------------------------------------------------------------------------------


# ASSEMBLY OF MASS MATRICIES -------------------------------------------------------------------------------------------
# def assembleMass(int K_Omega, int nE_Omega, long [:,:] Triangles, double [:,:] Verts, double [:,:] py_P, double [:] dx):
#     py_Ad = np.zeros(K_Omega**2).flatten("C")
#     cdef:
#         int nP = py_P.shape[0] # Does not differ in inner and outer integral!
#         long [:] c_Triangles = (np.array(Triangles, int)).flatten("C")
#         double [:] c_Verts = (np.array(Verts, float)).flatten("C")
#         double[:] P = (np.array(py_P, float)).flatten("C")
#         double[:] Ad = py_Ad
#
#     par_assembleMass(&Ad[0], &c_Triangles[0], &c_Verts[0], K_Omega, nE_Omega, nP, &P[0], &dx[0])
#     return np.reshape(py_Ad, (K_Omega, K_Omega))
#
# cdef class clsEvaluateMass:
#     cdef:
#         long K_Omega, nE_Omega, nP
#         long[:] c_Triangles
#         double[:] c_Verts
#         double[:] P
#         double [:] dx
#
#     def __init__(self, mesh, K_Omega, nE_Omega, Triangles, Verts, py_Px, dx, py_Py, dy):
#         self.K_Omega = mesh.K_Omega
#         self.nE_Omega = mesh.nE_Omega
#         self.c_Triangles = (np.array(mesh.triangles, int)).flatten("C")
#         self.c_Verts = (np.array(mesh.vertices, float)).flatten("C")
#         self.nPx = py_Px.shape[0]
#         self.Px = (np.array(py_Px, float)).flatten("C")
#         self.dx = dx
#         self.nPy = py_Py.shape[0]
#         self.Py = (np.array(py_Py, float)).flatten("C")
#         self.dy = dy
#         pass
#     def __call__(self, double [:] ud):
#         py_vd = np.zeros(self.K_Omega).flatten("C")
#         cdef:
#             double[:] vd = py_vd
#         par_evaluateMass(&vd[0], &ud[0], &self.c_Triangles[0], &self.c_Verts[0], self.K_Omega, self.nE_Omega, self.nP, &self.P[0], &self.dx[0])
#         return py_vd
# [END] ASSEMBLY OF MASS MATRICIES -------------------------------------------------------------------------------------------

# CLASS FOR Evaluation -------------------------------------------------------------------------------------------
# cdef class clsEvaluate:
#     cdef:
#         int K, K_Omega, nE, nE_Omega, nV, nV_Omega, nP
#         double sqdelta
#         double [:] P
#         double [:] dx
#         double [:] dy
#         long [:] c_Triangles
#         double [:] c_Verts
#         long [:] Neighbours
#
#     def __init__(self,
#                  # Mesh information ------------------------------------
#                  int K, int K_Omega,# Number of Basis functions
#                  int nE, int nE_Omega,# Number of Triangles and number of Triangles in Omega
#                  int nV, int nV_Omega, # Number of vertices (in case of CG = K and K_Omega)
#                  # Map Triangle index (Tdx) -> index of Vertices in Verts (Vdx = Triangle[Tdx] array of int, shape (3,))
#                  long [:,:] Triangles,
#                   # Map Vertex Index (Vdx) -> Coordinate of some Vertex i of some Triangle (E[i])
#                  double [:,:] Verts,
#                  double [:,:] py_P, # Quadrature Points
#                  double [:] dx,
#                  double [:] dy,
#                  double delta
#                  ):
#         self.K = K
#         self.K_Omega = K_Omega
#         self.nE = nE
#         self.nE_Omega = nE_Omega
#         self.nV = nV
#         self.nV_Omega = nV_Omega
#         self.sqdelta = pow(delta,2) # Squared interaction horizon
#
#         self.c_Triangles = (np.array(Triangles, int)).flatten("C")
#         self.c_Verts = (np.array(Verts, float)).flatten("C")
#
#         self.P = (np.array(py_P, float)).flatten("C")
#         self.nP = py_P.shape[0] # Does not differ in inner and outer integral!
#         self.dx = dx  # Weights for quadrature rule
#         self.dy = dy
#
#         # nVist of neighbours of each triangle, Neighbours[Tdx] returns row with Neighbour indices
#         self.Neighbours = self.nE*np.ones((self.nE*4), dtype=int)
#
#         # Setup adjaciency graph of the mesh --------------------------
#         neigs = []
#         for aTdx in range(self.nE):
#             neigs = get_neighbour(self.nE, &self.c_Triangles[0], &self.c_Triangles[4*aTdx])
#             n = len(neigs)
#             for i in range(n):
#                 self.Neighbours[4*aTdx + i] = neigs[i]
#
#     def __call__(self,
#             # Input vector
#             double [:] ud # Length K_Omega!
#         ):
#
#         ## Data Matrix ----------------------------------------
#         # Evaluates A_Omega only, because the input is truncated
#         py_vd = np.zeros(self.K_Omega).flatten("C")
#
#         cdef:
#             int aTdx=0, i=0
#             # Cython interface of C-aligned arrays of solution and right side
#             double[:] vd = py_vd
#
#         start = time.time()
#         # Compute Assembly
#         par_evaluateA(&ud[0], &vd[0], self.K, &self.c_Triangles[0], &self.c_Verts[0], self.nE , self.nE_Omega,
#                      self.nV, self.nV_Omega, self.nP, &self.P[0], &self.dx[0], &self.dy[0], self.sqdelta, &self.Neighbours[0])
#         total_time = time.time() - start
#         py_vd *= 2
#         return py_vd
#
#     def get_f(self):
#         py_fd = np.zeros(self.K_Omega).flatten("C")
#         cdef double[:] fd = py_fd
#         par_assemblef(&fd[0], &self.c_Triangles[0], &self.c_Verts[0], self.nE_Omega, self.nV_Omega, self.nP, &self.P[0], &self.dx[0])
#         return py_fd
# [END] CLASS FOR Evaluation -------------------------------------------------------------------------------------------