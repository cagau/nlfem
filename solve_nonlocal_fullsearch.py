#-*- coding:utf-8 -*-

import readmesh as rm
import numpy as np
import time
from quadpoints import P, weights
# Help Functions to check whether two elements interact
from nlocal import clsMesh, inNbhd, xinNbhd, clsInt
from aux import timestamp
# Necessary definitions for intersection -------------------------------------------------------------------------------

if __name__ == "__main__":
    # Nodes and Function values on nodes
    # Die Reihenfolge der phis muss zu der Reihenfolge
    # aus readmesh passen!

    # Mesh construction --------------------
    msh_name = "medium"
    args = rm.read_mesh("circle_" + msh_name + ".msh")
    mesh = clsMesh(*args)

    # Allocate Matrix A and right side f
    Ad = np.zeros((mesh.K_Omega, mesh.K))
    fd = np.zeros(mesh.K_Omega)
    delta = .2
    integrate = clsInt(P, weights, delta)

    # Loop over triangles --------------------------------------------------------------------------------------------------
    start = time.time()

    for aTdx in range(0, mesh.J_Omega): # Laufe über 0 bis KT_Omega (der Index der Dreiecke in Omega).
        aT = mesh[aTdx]
        # Get the indices of the nodes wrt. T.E (ak) and Verts (aK) which lie in Omega.
        aBdx_O, aVdx_O = mesh.Vdx_inOmega(aTdx)
        aVdx = mesh.Vdx(aTdx)  # Index for both Omega or OmegaI
        fd[aVdx_O] = integrate.f(aBdx_O, aT)

        queue = [aTdx]
        visited = np.array([False]*mesh.J)
        visited[aTdx] = True

        while queue != []:
            sTdx = queue.pop(0)
            NTdx = mesh.neighbor(sTdx)

            for bTdx in NTdx:
                if not visited[bTdx]:
                    bT = mesh[bTdx]
                    Mis_interact = inNbhd(aT, bT, delta, method="Ml2")

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

    total_time = time.time() - start

    Ad_O = Ad[:, :mesh.K_Omega]
    ud = np.linalg.solve(Ad_O, fd)

    fd_Ext = np.zeros(mesh.K)
    fd_Ext[:mesh.K_Omega] = fd
    ud_Ext = np.zeros(mesh.K)
    ud_Ext[:mesh.K_Omega] = ud

    tmstp = timestamp()
    np.savez(tmstp + "_" + msh_name + str(round(delta*10)), Ad_O, fd, ud_Ext, fd_Ext)

    print("Time needed", total_time)