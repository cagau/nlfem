import numpy as np

from conf import P, weights, delta, outerIntMethod, innerIntMethod
from nlocal import clsInt # Mesh, Triangle and Integrator class
from nbhd import inNbhd # Check interaction
import time

def assemble(mesh):
    """**Assembly routine**

    :param mesh: clsFEM. Mesh containing the data. All other data is read from *conf.py*.
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
        aPsidx, aAdx_O = mesh.Adx_inOmega(aTdx)
        aAdx = mesh.Adx(aTdx)  # Index for both Omega or OmegaI
        fd[aAdx_O] += integrate.f(aPsidx, aT)

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
                        bAdx = mesh.Adx(bTdx)

                        for i, aAdxi in enumerate(aAdx_O):
                            a = aPsidx[i]
                            for b in range(3):
                                termLocal, termNonloc = integrate.A(a, b, aT, bT, is_allInteract=Mis_interact.all())
                                Ad[aAdxi, aAdx[b]] += termLocal
                                Ad[aAdxi, bAdx[b]] -= termNonloc
            visited[NTdx] = True
        print("aTdx: ", aTdx, "\t Neigs: ", np.sum(visited), "\t Progress: ", round(aTdx / mesh.J_Omega * 100), "%", end="\r", flush=True)
    print("aTdx: ", aTdx, "\t Neigs: ", np.sum(visited), "\t Progress: ", round(aTdx / mesh.J_Omega * 100), "%\n")
    total_time = time.time() - start
    print("Time needed", total_time)
    Ad = 2*Ad

    return Ad, fd
