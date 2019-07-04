#-*- coding:utf-8 -*-

import readmesh as rm
import numpy as np
import time
# Help Functions to check whether two elements interact
from nlocal import clsMesh, inNbhd, xinNbhd, clsInt
from aux import timestamp
# Necessary definitions for intersection -------------------------------------------------------------------------------

if __name__ == "__main__":
    # Nodes and Function values on nodes
    # Die Reihenfolge der phis muss zu der Reihenfolge
    # aus readmesh passen!

    # Allocate Matrix A and right side f
    # Mesh construction --------------------

    msh_name = "medium"
    args = rm.read_mesh("circle_" + msh_name + ".msh")
    mesh = clsMesh(*args)
    delta = .2
    integrate = clsInt(mesh, delta)

    # Loop over triangles --------------------------------------------------------------------------------------------------
    start = time.time()

    for aTdx in range(0, mesh.J_Omega): # Laufe über 0 bis KT_Omega (der Index der Dreiecke in Omega).
        aT = mesh[aTdx]
        integrate.f(aTdx)

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
                        if Mis_interact.all():
                            integrate.A(aTdx, bTdx)
                        elif Mis_interact.any():
                            integrate.A(aTdx, bTdx, is_allInteract=False)
            visited[NTdx] = True

    total_time = time.time() - start

    Ad_O = integrate.Ad[:, :mesh.K_Omega]
    ud = np.linalg.solve(Ad_O, integrate.fd)

    fd_Ext = np.zeros(mesh.K)
    fd_Ext[:mesh.K_Omega] = integrate.fd
    ud_Ext = np.zeros(mesh.K)
    ud_Ext[:mesh.K_Omega] = ud

    tmstp = ""#timestamp()
    np.savez(tmstp + "_" + msh_name + str(round(delta*10)), Ad_O, integrate.fd, ud_Ext, fd_Ext)

    print("Time needed", total_time)