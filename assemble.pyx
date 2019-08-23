#cython: language_level=3
import numpy as np
from conf import py_P, py_weights, delta, outerIntMethod, innerIntMethod
from nlocal import A, f
from nlocal import inNbhd # Check interaction
import time
from nlocal import set_neighbour

def assemble(mesh):
    """**Assembly routine**

    :param mesh: clsFEM. Mesh containing the data. All other data is read from *conf.py*.
    :return: list of nd.array. Returns discretized matrix A and right side f.
    """

    # Allocate Matrix A and right side f
    Ad = np.zeros((mesh.K_Omega, mesh.K))
    fd = np.zeros(mesh.K_Omega)

    # Define Basis
    psi0 = 1 - py_P[0, :] - py_P[1, :]
    psi1 = py_P[0, :]
    psi2 = py_P[1, :]
    py_psi = np.array([psi0, psi1, psi2])

    # Define Data
    Verts = mesh.V
    Triangles = mesh.T
    J = mesh.J
    J_Omega = mesh.J_Omega
    L = mesh.L
    L_Omega = mesh.L_Omega

    # Import scipy sparse. Construct neighbourhood
    Neighbours = []

    for aTdx in range(J):
        Vdx = Triangles[aTdx]
        Neighbours.append(set_neighbour(Triangles, Vdx))

    # Fast Row Slicing
    #integrate = clsInt(py_P, py_weights, delta, outerIntMethod=outerIntMethod, innerIntMethod=innerIntMethod)
    cdef:
        double [:] dx = py_weights
        double [:] dy = py_weights
        double [:,:] P = py_P
        int nP = P.shape[1]
        double sqdelta = delta**2
        double [:,:] psi = py_psi

    Mis_interact = np.zeros(3)

    # Loop over triangles --------------------------------------------------------------------------------------------------
    start = time.time()
    for aTdx in range(0, mesh.J_Omega): # Laufe über 0 bis KT_Omega (der Index der Dreiecke in Omega).

        aTE = Verts[Triangles[aTdx]]
        # Get the indices of the nodes wrt. T.E (ak) and Verts (aK) which lie in Omega.
        ## aBdx should be called aPsidx !
        aPsidx, aAdx_O = mesh.Adx_inOmega(aTdx)
        aAdx = mesh.Adx(aTdx)  # Index for both Omega or OmegaI
        termf = np.zeros(3)
        # integrate over all elements
        f(aTE, P, nP, dx, psi, termf)
        # then assign to fd
        fd[aAdx_O] += termf[aPsidx]

        queue = [aTdx]
        visited = np.array([False]*mesh.J)
        while queue != []:
            sTdx = queue.pop(0)
            NTdx =  Neighbours[sTdx]
            for bTdx in NTdx:
                if not visited[bTdx]:
                    bTE = Verts[Triangles[bTdx]]
                    inNbhd(aTE, bTE, delta**2, Mis_interact)
                    if Mis_interact.any():
                        queue.append(bTdx)
                        bAdx = mesh.Adx(bTdx)
                        termLocal = np.zeros((3,3))
                        termNonloc = np.zeros((3,3))
                        A(aTE, bTE, P, nP, dx, dy, psi, sqdelta, Mis_interact.all(), termLocal, termNonloc)
                        for i, aAdxi in enumerate(aAdx_O):
                            a = aPsidx[i]
                            for b in range(3):
                                Ad[aAdxi, aAdx[b]] += termLocal[a][b]
                                Ad[aAdxi, bAdx[b]] -= termNonloc[a][b]
            visited[NTdx] = True
        print("aTdx: ", aTdx, "\t Neigs: ", np.sum(visited), "\t Progress: ", round(aTdx / mesh.J_Omega * 100),
              "%", end="\r", flush=True)
    print("aTdx: ", aTdx, "\t Neigs: ", np.sum(visited), "\t Progress: ", round(aTdx / mesh.J_Omega * 100), "%\n")
    total_time = time.time() - start
    print("Time needed", "{:1.2e}".format(total_time), " Sec")
    Ad = 2*Ad

    return Ad, fd
