#-*- coding:utf-8 -*-
# distutils: include_dirs = ../include

#cython: language_level=3
#cython: boundscheck=True, wraparound=True, cdivision=False
# Setting this compiler directive will given a minor increase of speed.

# C Includes (compile time)
from libcpp.string cimport string
cimport numpy as c_np
from libc.math cimport pow

# My Includes (compile time)
cimport Cassemble
cimport MeshTypes
cimport MeshType
cimport QuadratureType
cimport ConfigurationType

# Python includes (run time)
import numpy as np
import scipy.sparse as sparse
import time

# Assembly routines ####################################################################################################
def stiffnessMatrix(
    mesh,
    kernel,
    configuration
):
    # Quadrature
    dim = mesh.get("vertices", ValueError("No vertices provided")).shape[1]
    cquadrature = CQuadrature(dim, configuration)

    # Mesh
    cmesh = CMesh(mesh)

    # ConfigurationType
    cconf = CConfiguration(kernel, configuration)

    Cassemble.stiffnessMatrix(cmesh.Cmesh[0], cquadrature.Cquadrature[0], cconf.Cconf[0])



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
    Warning("This function is deprecated and will soone be deleted.")
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

# Class Wrappers #######################################################################################################
cdef class CQuadrature:
    cdef QuadratureType.QuadratureType * Cquadrature

    def __cinit__(self, dim, configuration):
        # Quadrature Rules
        cdef double [:] Px = configuration["quadrature"]["outer"]["points"].flatten()
        cdef long nPx = configuration["quadrature"]["outer"]["points"].shape[0]
        cdef double [:] Py = configuration["quadrature"]["inner"]["points"].flatten()
        cdef long nPy = configuration["quadrature"]["inner"]["points"].shape[0]
        dx_ = configuration["quadrature"]["outer"]["weights"].flatten()
        cdef double [:] dx = dx_
        dy_ = configuration["quadrature"]["inner"]["weights"].flatten()
        cdef double [:] dy = dy_

        if (nPx != len(dx_)) or (nPy != len(dy_)):
            raise ValueError("Quadrature Points should be of shape (n quadrature points, dimension).")

        cdef double [:] Pg
        cdef const double * ptrPg = NULL
        cdef double [:] dg
        cdef const double * ptrdg = NULL

        cdef long tensorGaussDegree = configuration.get("tensorGaussDegree", 0)
        if tensorGaussDegree:
            quadgauss = tensorgauss(tensorGaussDegree)
            Pg = quadgauss.points.flatten()
            ptrPg = &Pg[0]
            dg = quadgauss.weights.flatten()
            ptrdg = &dg[0]

        self.Cquadrature = new QuadratureType.QuadratureType(dim, &Px[0], &dx[0], nPx,
                                                &Py[0], &dy[0], nPy, ptrPg, ptrdg,
                                                tensorGaussDegree)


cdef class CMesh:
    cdef MeshType.MeshType * Cmesh

    def __cinit__(self, mesh, long outdim):
        # Mesh
        cdef double maxDiameter = mesh.get("maxDiameter", 0.0)
        cdef long dim = mesh.get("vertices", ValueError("No vertices provided")).shape[1]

        cdef double[:] vertices = mesh.get("vertices").flatten()
        elements_ = mesh.get("elements", ValueError("No elements provided"))
        #neighbors = constructAdjaciencyCSRGraph(elements_)
        neighbors = constructAdjaciencyGraph(elements_)
        cdef long nNeighbours = neighbors.shape[1]
        cdef long[:] ptrNeighborIndices = np.array(neighbors.indices, dtype=np.int)
        cdef long[:] ptrNeighborIndexPtr = np.array(neighbors.indptr, dtype=np.int)
        cdef long[:] elements = elements_.flatten()

        elementLabels = mesh.get("elementLabels", ValueError("No elementLabels provided"))
        elementLabels = sparse.csr_matrix(elementLabels)
        cdef long[:] elementLabelsData = elementLabels.data
        cdef long[:] elementLabelsIndices = np.array(elementLabels.indices, dtype=np.int)
        cdef long[:] elementLabelsIndptr = np.array(elementLabels.indptr, dtype=np.int)
        cdef long nE = elementLabels.shape[0]
        cdef long nEOmega = np.sum(elementLabels.data > 0)

        vertexLabels = mesh.get("vertexLabels", ValueError("No vertexLabels provided"))
        vertexLabels = sparse.csr_matrix(vertexLabels, dtype=np.float)
        cdef double[:] vertexLabelsData = vertexLabels.data
        cdef long[:] vertexLabelsIndices = np.array(vertexLabels.indices, dtype=np.int)
        cdef long[:] vertexLabelsIndptr = np.array(vertexLabels.indptr, dtype=np.int)
        cdef long nV = vertexLabels.shape[0]
        cdef long nVOmega = np.sum(vertexLabels.data > 0)

        self.Cmesh = new MeshType.MeshType(&elements[0], &elementLabelsData[0], &vertices[0],
                        nE, nEOmega, nV, nVOmega, &ptrNeighborIndices[0], &ptrNeighborIndexPtr[0],
                        dim, maxDiameter, outdim, nNeighbours)

cdef class CConfiguration:
    cdef ConfigurationType.ConfigurationType * Cconf

    def __cinit__(self, kernel, configuration):
        # Model
        cdef string kernelFunction = kernel.get("function", ValueError("No kernel function given.")).encode('UTF-8')
        cdef double kernelHorizon = kernel.get("horizon", ValueError("No interaction horzon function defined."))

        # Approx Balls
        cdef string integrationMethod = configuration["approxBalls"]["method"].encode('UTF-8')
        cdef int isPlacePointOnCap = configuration["approxBalls"].get("isPlacePointOnCap", True)
        cdef double [:]  averageBallWeights = np.array(configuration["approxBalls"].get("averageBallWeights", [0.,1.,1.]))

        # Ansatz
        cdef int isDiscontinuousGalerkin = configuration["ansatz"] == "DG"
        # outputdim -> attribute of the kernel (which we don't know here!)
        # It should not be possible to mix matrix and scalar kernels though!
        # -> The kernel has only ONE attribute outdim, even if label dependent
        # K, K_Omega -> comes after the kernel, and ansatz
        cdef long K=0, K_Omega=0 ### ???
        # Save Path
        cdef string path_stiffnesMatrix = configuration.get("savePath", "_tmpSavePath_stiffnesMatrix_").encode('UTF-8')

        self.Cconf = new ConfigurationType.ConfigurationType(path_stiffnesMatrix, "".encode('UTF-8'),
                  kernelFunction, "".encode('UTF-8'),
                  integrationMethod,
                  isPlacePointOnCap, kernelHorizon)

cdef class Element:
    cdef MeshTypes.ElementClass element

    def __cinit__(self, int dim):
        self.element = MeshTypes.ElementClass(dim)

def showElement(int dim):
    E = Element(dim)
    return MeshTypes.getElement(E.element)

# Helper functions #####################################################################################################
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

# Numerical functions ##################################################################################################
def constructAdjaciencyCSRGraph(elements, verbose = False):
    if verbose: print("Constructing adjaciency graph...")
    nE = elements.shape[0]
    nV = np.max(elements)+1
    dVerts = elements.shape[1]
    dim = dVerts-1
    cdef int ncommon = 1

    indptr = np.zeros(nE + 1)
    indptr[:nE] = np.arange(0,nE*dVerts,dVerts)
    indptr[nE] = nE*dVerts

    indices = elements.flatten()
    data = np.ones(len(indices))

    grph_elements_csr = sparse.csr_matrix((data, indices, indptr), shape = (nE, nV))
    grph_elements_csc = sparse.csc_matrix((data, indices, indptr), shape = (nV, nE))
    if verbose: print("Done [Constructing adjaciency graph]")
    return sparse.csr_matrix((grph_elements_csr @ grph_elements_csc) >= ncommon, dtype=np.int64)

def constructAdjaciencyGraph(long[:,:] elements):
    Warning("This function is deprecated and will soone be deleted.")
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
    Warning("This function is deprecated and will soone be deleted.")
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
