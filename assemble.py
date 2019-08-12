import numpy as np

from conf import P, weights, delta, outerIntMethod, innerIntMethod
from nlocal import clsInt # Mesh, Triangle and Integrator class
from nbhd import inNbhd # Check interaction
import time

def assemble(mesh):
    """**Assembly routine**

    :param mesh: clsMesh. Mesh containing the data. All other data is read from *conf.py*.
    :return: list of nd.array. Returns discretized matrix A and right side f.
    """

    # Allocate Matrix A and right side f
    Ad = np.zeros((mesh.K_Omega, mesh.K))
    fd = np.zeros(mesh.K_Omega)
    integrate = clsInt(P, weights, delta, outerIntMethod=outerIntMethod, innerIntMethod=innerIntMethod)

    # Loop over triangles --------------------------------------------------------------------------------------------------
    start = time.time()
    for aTdx in range(0, mesh.J_Omega): # Laufe über 0 bis KT_Omega (der Index der Dreiecke in Omega).

        aT = mesh[aTdx]
        # Get the indices of the nodes wrt. T.E (ak) and Verts (aK) which lie in Omega.
        ## aBdx should be called aPsidx !
        aBdx_O, aVdx_O = mesh.Vdx_inOmega(aTdx)
        aVdx = mesh.Vdx(aTdx)  # Index for both Omega or OmegaI
        fd[aVdx_O] += integrate.f(aBdx_O, aT)

        queue = [aTdx]
        visited = np.array([False]*mesh.J)

        while queue != []:
            sTdx = queue.pop(0)
            NTdx = mesh.neighbor(sTdx)

            for bTdx in NTdx:
                if not visited[bTdx]:
                    bT = mesh[bTdx]
                    Mis_interact = inNbhd(aT, bT, delta, method="Ml2Bary")
                    if Mis_interact.any():
                        queue.append(bTdx)
                        bVdx = mesh.Vdx(bTdx)

                        for i, avdx in enumerate(aVdx_O):
                            a = aBdx_O[i]
                            for b in range(3):
                                termLocal, termNonloc = integrate.A(a, b, aT, bT, is_allInteract=Mis_interact.all())
                                Ad[avdx, aVdx[b]] += termLocal
                                Ad[avdx, bVdx[b]] -= termNonloc
            visited[NTdx] = True
        print("aTdx: ", aTdx, "\t Neigs: ", np.sum(visited), "\t Progress: ", round(aTdx / mesh.J_Omega * 100), "%", end="\r", flush=True)
    print("aTdx: ", aTdx, "\t Neigs: ", np.sum(visited), "\t Progress: ", round(aTdx / mesh.J_Omega * 100), "%\n")
    total_time = time.time() - start
    print("Time needed", total_time)
    Ad = 2*Ad

    return Ad, fd
