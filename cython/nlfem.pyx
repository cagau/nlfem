#-*- coding:utf-8 -*-
# distutils: include_dirs = ../include

#cython: language_level=3
#cython: boundscheck=True, wraparound=True, cdivision=False
# Setting this compiler directive will given a minor increase of speed.

# Assembly routine
from libcpp.string cimport string
cimport Cassemble
cimport MeshTypes

import numpy as np
import time
from libc.math cimport pow
import scipy.sparse as sparse
cimport numpy as c_np

cdef class Element:
    cdef MeshTypes.ElementClass element

    def __cinit__(self, int dim):
        self.element = MeshTypes.ElementClass(dim)

def showElement(int dim):
    E = Element(dim)
    return MeshTypes.getElement(E.element)

def read_arma_mat(path, is_verbose=False):
    """
    Read armadillo vector format from file.

    :param path: string, Path to file.
    :param is_verbose: bool, Verbose mode.
    :return: np.ndarray, Vector of double.
    """
    import numpy as np

    sizeof_double = 8

    f = open(path, "rb")
    # Read Armadillo header
    arma_header = f.readline()
    if arma_header != b'ARMA_MAT_BIN_FN008\n':
        raise ValueError("in read_arma_mat(), input file is of wrong format.")
    # Get shape of sparse matrix
    arma_shape = f.readline()
    n_rows, n_cols = tuple([int(x) for x in arma_shape.decode("utf-8").split()])
    if is_verbose: print("Shape (", n_rows, ", ", n_cols, ")", sep="")
    # Raw binary of sparse Matrix in csc-format
    b_data = f.read()
    f.close()

    b_values = b_data[:sizeof_double * n_rows * n_cols]
    values = np.array(np.frombuffer(b_values)).reshape((n_rows, n_cols), order="F")
    if is_verbose: print("Values ", values)
    if is_verbose: print(values)

    return values

def read_arma_spMat(path, is_verbose=False):
    """
    Read sparse Matrix in armadillo spMat format from file.

    :param path: string, Path to file.
    :param is_verbose: bool, Verbose mode.
    :return: scipy.csr_matrix, Matrix of double.
    """

    import scipy.sparse as sp
    import numpy as np

    sizeof_double = 8

    f = open(path, "rb")
    # Read Armadillo header
    arma_header = f.readline()
    if arma_header != b'ARMA_SPM_BIN_FN008\n':
        raise ValueError("in read_arma_spMat(), input file is of wrong format.")
    # Get shape of sparse matrix
    arma_shape = f.readline()
    n_rows, n_cols, n_nonzero = tuple([int(x) for x in arma_shape.decode("utf-8").split()])
    if is_verbose: print("Shape (", n_rows, ", ", n_cols, ")", sep="")
    # Raw binary of sparse Matrix in csc-format
    b_data = f.read()
    b_values = b_data[:sizeof_double * n_nonzero]
    b_pointers = b_data[sizeof_double * n_nonzero:]
    f.close()

    values = np.frombuffer(b_values)
    if is_verbose: print("Values ", values)

    pointers = np.frombuffer(b_pointers, dtype=np.uint)
    row_index = pointers[:n_nonzero]
    if is_verbose: print("Row index", row_index)
    col_pointer = pointers[n_nonzero:]
    if is_verbose: print("Column pointer", col_pointer)

    A = sp.csc_matrix((values, row_index, col_pointer), shape=(n_rows, n_cols)).transpose()
    A = A.tocsr() # This is efficient, linearly in n_nonzeros.
    if is_verbose: print(A.todense())
    return A

def remove_arma_tmp(path):
    import os
    os.remove(path)

# Tensor Gauss Quadrature
class tensorgauss:
    def __init__(self, deg, dim=4):
        self.deg = deg
        self.dim = dim
        p, w = np.polynomial.legendre.leggauss(deg)
        #print("Length", len(p), "\np", p, "w", w)
        self.N = deg**dim

        self.weights = np.ones(deg**dim)
        self.points = np.zeros((deg**dim, dim))
        l = 0
        k = np.zeros(self.dim, dtype=int)

        point_coord = np.array([row.flatten() for row in np.meshgrid(*([np.arange(self.deg)]*self.dim),
        indexing = "ij")]).T
        for k, dxRow in enumerate(point_coord):
            self.points[k] = p[dxRow]
            for dx in dxRow:
                self.weights[k] *= w[dx]

def stiffnesMatrix(
    mesh,
    kernel,
    configuration
):

    # Save Path
    cdef string path_stiffnesMatrix = configuration.get("savePath", "_tmpSavePath_stiffnesMatrix_").encode('UTF-8')

    # Model
    cdef string kernelFunction = kernel.get("function", ValueError("No kernel function given.")).encode('UTF-8')
    cdef double kernelHorizon = kernel.get("horizon", ValueError("No interaction horzon function defined."))

    # Quadrature Rules
    cdef double [:] Px = configuration["outer"]["points"].flatten()
    cdef double nPx = configuration["outer"]["points"].shape[0]
    cdef double [:] Py = configuration["inner"]["points"].flatten()
    cdef double nPy = configuration["inner"]["points"].shape[0]
    cdef double [:] dx = configuration["outer"]["weights"].flatten()
    cdef double [:] dy = configuration["inner"]["weights"].flatten()

    if (nPx != len(dx)) or (nPy != len(dy)):
        raise ValueError("Quadrature Points should be of shape (n quadrature points, dimension).")

    cdef double [:] Pg
    cdef const double * ptrPg = NULL
    cdef double [:] dg
    cdef const double * ptrdg = NULL

    tensorGaussDegree = configuration.get("tensorGaussDegree", {})
    if tensorGaussDegree:
        quadgauss = tensorgauss(tensorGaussDegree)
        Pg = quadgauss.points.flatten()
        ptrPg = &Pg[0]
        dg = quadgauss.weights.flatten()
        ptrdg = &dg[0]

    # Mesh
    cdef double maxDiameter = mesh.get("maxDiameter", 0.0)

    cdef double[:] vertices = mesh.get("vertices", ValueError("No vertices provided"))
    elements = mesh.get("elements", ValueError("No elements provided"))
    cdef long[:] neighbors = constructAdjaciencyGraph(elements)
    cdef long[:] elements = mesh.get("elements", ValueError("No elements provided"))

    elementLabels = mesh.get("elementLabels", ValueError("No elementLabels provided"))
    elementLabels = sparse.csr_matrix(elementLabels, dtype=np.int)
    elementLabelsData = elementLabels.data
    cdef long[:] elementLabelsData = elementLabels.data
    cdef long[:] elementLabelsIndices = elementLabels.indices
    cdef long[:] elementLabelsIndptr = elementLabels.indptr
    cdef long nE = elementLabels.shape[0]
    cdef long nEOmega = np.sum(elementLabelsData > 0)

    vertexLabels = mesh.get("vertexLabels", ValueError("No vertexLabels provided"))
    vertexLabels = sparse.csr_matrix(vertexLabels, dtype=np.float)
    vertexLabels = vertexLabels.data
    cdef long[:] vertexLabelsData = vertexLabels.data
    cdef long[:] vertexLabelsIndices = vertexLabels.indices
    cdef long[:] vertexLabelsIndptr = vertexLabels.indptr
    cdef long nV = vertexLabels.shape[0]
    cdef long nVOmega = np.sum(vertexLabelsData > 0)


    # Things which apparently are NOT set here

    # outputdim -> attribute of the kernel (which we don't know here!)
    # It should not be possible to mix matrix and scalar kernels though!
    # -> The kernel has only ONE attribute outdim, even if label dependent
    # K, K_Omega -> comes after the kernel, and ansatz

    start = time.time()
    Cassemble.par_assemble( "system".encode('UTF-8'),
                            path_stiffnesMatrix,
                            NULL,
                            mesh.K_Omega,
                            mesh.K,
                            &elements[0], &elementLabels[0],
                            &vertices[0],
                            &verexLabels[0], ## NEW!!
                            nE , nEOmega,
                            nV, nVOmega,
                            &ptrPx[0], nPx, &ptrdx[0],
                            &ptrPy[0], nPy, &ptrdy[0],
                            delta**2,
                            &neighbors[0],
                            nNeighbours,
                            mesh.is_DiscontinuousGalerkin,
                            mesh.is_NeumannBoundary,
                            &model_kernel_[0],
                            &model_f_[0],
                            &integration_method_[0],
                            is_PlacePointOnCap_,
                            mesh.dim, outdim, ptrZetaIndicator, nZeta,
                            ptrPg, tensorGaussDegree, ptrdg, maxDiameter)

    total_time = time.time() - start
    print("Assembly Time\t", "{:1.2e}".format(total_time), " Sec")

    Ad = read_arma_spMat(path_spAd)
    if is_tmpAd:
        remove_arma_tmp(path_spAd)

def loadVector(
    mesh,
    load,
    configuration
):
    pass
def assemble(
        # Mesh information ------------------------------------
        mesh,
        Px,
        Py,
        # Weights for quadrature rule
        dx,
        dy,
        double delta,
        path_spAd=None,
        path_fd=None,
        compute="systemforcing", # "forcing", "system"
        model_kernel="constant",
        model_f = "constant",
        integration_method = "retriangulate",
        is_PlacePointOnCap = 1,
        tensorGaussDegree=0
    ):
    is_tmpAd = False
    if path_spAd is None:
        is_tmpAd = True
        path_spAd = "tmp_spAd"
    cdef string path_spAd_ = path_spAd.encode('UTF-8')
    Ad = None

    is_tmpfd = False
    if path_fd is None:
        is_tmpfd = True
        path_fd = "tmp_fd"
    cdef string path_fd_ = path_fd.encode('UTF-8')
    fd = None

    cdef long[:] neighbours = mesh.neighbours.flatten()#nE*np.ones((nE*dVertex), dtype=int)
    cdef int nNeighbours = mesh.neighbours.shape[1]
    cdef long[:] elements = mesh.elements.flatten()
    cdef long[:] elementLabels = mesh.elementLabels.flatten()
    cdef double[:] vertices = mesh.vertices.flatten()
    #cdef double[:] ptrAd = Ad
    cdef double[:] ptrfd = fd
    cdef string model_kernel_ = model_kernel.encode('UTF-8')
    cdef string model_f_ = model_f.encode('UTF-8')
    cdef string integration_method_ = integration_method.encode('UTF-8')

    cdef string compute_system_ = "system".encode('UTF-8')
    cdef string compute_forcing_ = "forcing".encode('UTF-8')
    cdef int is_PlacePointOnCap_ = is_PlacePointOnCap


    cdef double [:] ptrPx = Px.flatten()
    cdef double [:] ptrPy = Py.flatten()
    cdef double [:] ptrdx = dx.flatten()
    cdef double [:] ptrdy = dy.flatten()

    cdef long [:] ZetaIndicator
    cdef long * ptrZetaIndicator = NULL
    cdef long nZeta

    try:
        nZeta = mesh.ZetaIndicator.shape[1]
        ZetaIndicator = mesh.ZetaIndicator.flatten()
        if nZeta > 0:
            ptrZetaIndicator = &ZetaIndicator[0]
    except AttributeError:
        print("Zeta not found.")
        nZeta = 0

    try:
        outdim = mesh.outdim
    except AttributeError:
        print("Mesh out dim not found.")
        outdim = 1

    cdef double [:] Pg
    cdef const double * ptrPg = NULL
    cdef double [:] dg
    cdef const double * ptrdg = NULL

    cdef double maxDiameter = 0.0
    try:
        maxDiameter = mesh.diam
    except AttributeError:
        print("Element diameter not found.")

    if tensorGaussDegree != 0:
        quadgauss = tensorgauss(tensorGaussDegree)
        Pg = quadgauss.points.flatten()
        ptrPg = &Pg[0]
        dg = quadgauss.weights.flatten()
        ptrdg = &dg[0]

    # Compute Assembly --------------------------------
    if (compute=="system" or compute=="systemforcing"):
        start = time.time()
        Cassemble.par_assemble( compute_system_, path_spAd_, path_fd_, mesh.K_Omega, mesh.K,
                            &elements[0], &elementLabels[0], &vertices[0],
                            mesh.nE , mesh.nE_Omega, mesh.nV, mesh.nV_Omega,
                            &ptrPx[0], Px.shape[0], &ptrdx[0],
                            &ptrPy[0], Py.shape[0], &ptrdy[0],
                            delta**2,
                            &neighbours[0],
                            nNeighbours,
                            mesh.is_DiscontinuousGalerkin,
                            mesh.is_NeumannBoundary,
                            &model_kernel_[0],
                            &model_f_[0],
                            &integration_method_[0],
                            is_PlacePointOnCap_,
                            mesh.dim, outdim, ptrZetaIndicator, nZeta,
                            ptrPg, tensorGaussDegree, ptrdg, maxDiameter)

        total_time = time.time() - start
        print("Assembly Time\t", "{:1.2e}".format(total_time), " Sec")

        Ad = read_arma_spMat(path_spAd)
        if is_tmpAd:
            remove_arma_tmp(path_spAd)

    if (compute=="forcing" or compute =="systemforcing"):
        print("")
        Cassemble.par_assemble( compute_forcing_, path_spAd_, path_fd_, mesh.K_Omega, mesh.K,
                            &elements[0], &elementLabels[0], &vertices[0],
                            mesh.nE , mesh.nE_Omega, mesh.nV, mesh.nV_Omega,
                            &ptrPx[0], Px.shape[0], &ptrdx[0],
                            &ptrPy[0], Py.shape[0], &ptrdy[0],
                            delta**2,
                            &neighbours[0],
                            nNeighbours,
                            mesh.is_DiscontinuousGalerkin,
                            mesh.is_NeumannBoundary,
                            &model_kernel_[0],
                            &model_f_[0],
                            &integration_method_[0],
                            is_PlacePointOnCap_,
                            mesh.dim, outdim, ptrZetaIndicator, nZeta,
                            ptrPg, tensorGaussDegree, ptrdg, maxDiameter)

        fd = read_arma_mat(path_fd)[:,0]
        if is_tmpfd:
            remove_arma_tmp(path_fd)

    return Ad, fd

def evaluateMass(
      # Mesh information ------------------------------------
            mesh,
            ud,
            Px,
            # Weights for quadrature rule
            dx
        ):
    vd = np.zeros(mesh.K_Omega)
    cdef double[:] ptrvd = vd
    cdef double[:] ptrud = ud.flatten()

    cdef long[:] elements = mesh.elements.flatten()
    cdef long [:] elementLabels = mesh.elementLabels.flatten()
    cdef double[:] vertices = mesh.vertices.flatten()

    cdef double[:] ptrPx = Px.flatten()
    cdef double[:] ptrdx = dx.flatten()

    cdef long outdim = 1
    try:
        outdim = mesh.outdim
    except AttributeError:
        pass

    Cassemble.par_evaluateMass(
            &ptrvd[0],
            &ptrud[0],
            &elements[0],
            &elementLabels[0],
            &vertices[0],
            mesh.K_Omega,
            mesh.nE_Omega,
            Px.shape[0], &ptrPx[0], &ptrdx[0], mesh.dim, outdim)
    return vd



def constructAdjaciencyCSRGraph(long[:,:] elements):
    print("Constructing adjaciency graph...")
    nE = elements.shape[0]
    nV = np.max(elements)+1
    cdef int dVerts = elements.shape[1]
    cdef int dim = dVerts-1

    indptr = np.zeros(nE + 1)
    indptr[:nE] = np.arange(0,nE*3,3)
    indptr[nE] = nE*3
    indices = elements.ravel()
    data = np.ones(len(indices)-1)

    grph_elements_csr = sparse.csr_matrix((data, indices, indptr), shape = (nE, dVerts))
    grph_elements_csc = sparse.csc_matrix((data, indices, indptr), shape = (nE, dVerts))

    for Tdx, Vdx in enumerate(elements):
        for d in range(dVerts):
            grph_elements[Vdx[d], Tdx] = 1
    #grph_neigs = ((grph_elements.transpose() @ grph_elements) == dim)
    grph_neigs = ((grph_elements.transpose() @ grph_elements) > 0)

    # ..

    data = [1,2,3]
    indptr  = [0,1,1,3]
    index = [0,1,3]
    sparse.csr_matrix((data, index, indptr)).todense()

def constructAdjaciencyGraph(long[:,:] elements):
    print("Constructing adjaciency graph...")
    nE = elements.shape[0]
    nV = np.max(elements)+1
    cdef int dVerts = elements.shape[1]
    cdef int dim = dVerts-1

    #neigs = np.ones((nE, dim+1), dtype=np.int)*nE
    grph_elements = sparse.lil_matrix((nV, nE), dtype=np.int)

    for Tdx, Vdx in enumerate(elements):
        for d in range(dVerts):
            grph_elements[Vdx[d], Tdx] = 1
    #grph_neigs = ((grph_elements.transpose() @ grph_elements) == dim)
    grph_neigs = ((grph_elements.transpose() @ grph_elements) > 0)

    rowsum = np.sum(grph_neigs, axis= 0)
    nCols = np.max(rowsum)
    neigs = np.ones((nE, nCols), dtype=np.int)*nE
    elemenIndices, neighbourIndices = grph_neigs.nonzero()

    neigs[elemenIndices[0],0] = neighbourIndices[0]
    cdef int colj = 0
    cdef int k

    for k in range(1, len(elemenIndices)):
        colj *= ((elemenIndices[k-1]-elemenIndices[k])==0)
        colj += ((elemenIndices[k-1]-elemenIndices[k])==0)
        neigs[elemenIndices[k], colj] =  neighbourIndices[k]
    return neigs#, grph_neigs2

def solve_cg(Q, c_np.ndarray  b, c_np.ndarray x, double tol=1e-9, int max_it = 500):
    cdef int n = b.size
    cdef int k=0

    cdef double beta = 0.0
    cdef c_np.ndarray p = np.zeros(n)
    cdef c_np.ndarray r = Q.dot(x) - b
    cdef double res_new  = np.linalg.norm(r)

    while res_new >= tol and k < max_it:
        k+=1
        p = -r + beta*p
        alpha = res_new**2 / p.dot(Q.dot(p))
        x = x + alpha*p
        r = r + alpha*Q.dot(p)
        res_old = res_new
        res_new = np.linalg.norm(r)
        beta = res_new**2/res_old**2

    return {"x": x, "its": k, "res": res_new}

# DEBUG Helpers - -----------------------------------------------------------------------------------------------------
from Cassemble cimport method_retriangulate
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
    Rdx = method_retriangulate(&x_center[0], &TE[0], sqdelta, &cTriangleList[0], is_placePointOnCaps)

    return Rdx, TriangleList

from Cassemble cimport toRef
def py_toRef(
    double [:] TE,
    double [:] phys_x):
    ref_p = np.zeros(2)
    cdef double [:] cref_p = ref_p
     # void toRef(const double * E, const double * phys_x, double * ref_p);
    toRef(&TE[0], &phys_x[0], &cref_p[0]);
    return ref_p

from Cassemble cimport toPhys
def py_toPhys(
    double [:] TE,
    double [:] p):
    out_x = np.zeros(2)
    cdef double [:] cout_x = out_x
     # void toPhys(const double * E, const double * p, int dim, double * out_x)
    toPhys(&TE[0], &p[0], 2, &cout_x[0]);
    return out_x

from Cassemble cimport solve2x2
def py_solve2x2(
    double [:] A,
    double [:] b
    ):
    x = np.zeros(2)
    cdef double [:] cx = x
    # void solve2x2(const double * A, const double * b, double * x)
    solve2x2(&A[0], &b[0], &cx[0])
    return x
"""
from Cassemble cimport retriangulate
from Cassemble cimport toRef, model_basisFunction
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
        long[:] Neighbours = mesh.nE*np.ones((mesh.nE*dVertex), dtype=int)
        # Squared interaction horizon
        double sqdelta = pow(delta,2)

    # Setup adjaciency graph of the mesh --------------------------
    neigs = []
    for aTdx in range(mesh.nE):
        neigs = get_neighbour(mesh.nE, dVertex, &Triangles[0], &Triangles[(dVertex+1)*aTdx])
        n = len(neigs)
        for i in range(n):
            Neighbours[4*aTdx + i] = neigs[i]

    start = time.time()

    cdef int  h=0
    ## General Loop Indices ---------------------------------------
    cdef int bTdx=0

    ## Breadth First Search --------------------------------------
    cdef visited = np.zeros(mesh.nE)#(int *) malloc(nE*sizeof(int));

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
            aTE[2*0+j] = Verts[2*Triangles[(dVertex+1)*aTdx+1] + j]
            aTE[2*1+j] = Verts[2*Triangles[(dVertex+1)*aTdx+2] + j]
            aTE[2*2+j] = Verts[2*Triangles[(dVertex+1)*aTdx+3] + j]

        ## compute Determinant
        aTdet = absDet(aTE)
        labela = Triangles[(dVertex+1)*aTdx]

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
                ## In order to be able to store the list as contiguous array we fill up the empty spots with the number nE
                ## i.e. the total number of Triangles (which cannot be an index).
                if (bTdx < mesh.nE):

                    ## Prepare Triangle information bTE and bTdet ------------------
                    ## Copy coordinates of Triange b to bTE.
                    ## again this is done fore convenience only, actually those are unnecessary copies!
                    for i in range(2):
                        bTE[2*0+i] = Verts[2*Triangles[(dVertex+1)*bTdx+1] + i]
                        bTE[2*1+i] = Verts[2*Triangles[(dVertex+1)*bTdx+2] + i]
                        bTE[2*2+i] = Verts[2*Triangles[(dVertex+1)*bTdx+3] + i]

                    bTdet = absDet(bTE)
                    labelb = Triangles[(dVertex+1)*bTdx]
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

cdef toPhys(double [:] E, double * p, double * out_x):
    cdef int i=0
    for i in range(2):
        out_x[i] = (E[2*1+i] - E[2*0+i])*p[0] + (E[2*2+i] - E[2*0+i])*p[1] + E[2*0+i]

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

"""
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
#         self.Neighbours = self.nE*np.ones((self.nE*dVertex), dtype=int)
#
#         # Setup adjaciency graph of the mesh --------------------------
#         neigs = []
#         for aTdx in range(self.nE):
#             neigs = get_neighbour(self.nE, dVertex, &self.c_Triangles[0], &self.c_Triangles[4*aTdx])
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
