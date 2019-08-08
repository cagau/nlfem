#-*- coding:utf-8 -*-

from conf import P, weights, mesh_name, delta

import numpy as np
import time
import pickle as pkl

from nlocal import clsMesh, clsInt # Mesh, Triangle and Integrator class
from nbhd import inNbhd, xnotinNbhd # Check interaction
from aux import filename
from plot import plot

# Necessary definitions for intersection -------------------------------------------------------------------------------
if __name__ == "__main__":
    # Nodes and Function values on nodes
    # Die Reihenfolge der phis muss zu der Reihenfolge
    # aus readmesh passen!

    # Mesh construction --------------------
    mesh = clsMesh("circle_" + mesh_name + ".msh")
    print("Delta: ", delta, "\t Mesh: ", mesh_name)
    print("Number of basis functions: ", mesh.K)
    # Allocate Matrix A and right side f
    Ad = np.zeros((mesh.K_Omega, mesh.K))
    Ad_any = np.zeros((mesh.K_Omega, mesh.K))
    fd = np.zeros(mesh.K_Omega)
    integrate = clsInt(P, weights, delta)

    # Loop over triangles --------------------------------------------------------------------------------------------------
    start = time.time()
    for aTdx in range(0, mesh.J_Omega): # Laufe über 0 bis KT_Omega (der Index der Dreiecke in Omega).

        aT = mesh[aTdx]
        # Get the indices of the nodes wrt. T.E (ak) and Verts (aK) which lie in Omega.
        ## aBdx should be called aPsidx !
        aBdx_O, aVdx_O = mesh.Vdx_inOmega(aTdx)
        aVdx = mesh.Vdx(aTdx)  # Index for both Omega or OmegaI
        fd[aVdx_O] = integrate.f(aBdx_O, aT)

        queue = [aTdx]
        visited = np.array([False]*mesh.J)
        boundary = np.array([False]*mesh.J)
        #visited[aTdx] = True

        while queue != []:
            sTdx = queue.pop(0)
            NTdx = mesh.neighbor(sTdx)

            for bTdx in NTdx:
                if not visited[bTdx]:
                    bT = mesh[bTdx]
                    Mis_interact = inNbhd(aT, bT, delta, method="Ml2Bary")
                    if Mis_interact.all():
                        queue.append(bTdx)
                        bVdx = mesh.Vdx(bTdx)

                        for i, avdx in enumerate(aVdx_O):
                            a = aBdx_O[i]
                            for b in range(3):
                                if Mis_interact.all():
                                    termLocal, termNonloc = integrate.A(a, b, aT, bT, is_allInteract=Mis_interact.all())
                                    Ad[avdx, aVdx[b]] += termLocal
                                    Ad[avdx, bVdx[b]] -= termNonloc
                                else:
                                    termLocal, termNonloc = integrate.A(a, b, aT, bT, is_allInteract=Mis_interact.all())
                                    Ad_any[avdx, aVdx[b]] += termLocal
                                    Ad_any[avdx, bVdx[b]] -= termNonloc
            visited[NTdx] = True
        #all_NTdx = list(np.flatnonzero(visited))
        #mesh.plot([aTdx] + all_NTdx, is_plotmsh=True, pdfname="output/breadthTest/"+str(aTdx)+"_all", delta=delta, title=str(aTdx), refPoints=P)
        print("aTdx: ", aTdx, "\t Neigs: ", np.sum(visited), "\t Progress: ", round(aTdx / mesh.J_Omega * 100), "%", end="\r", flush=True)
    total_time = time.time() - start
    #Ad *= 2
    Ad += Ad_any
    Ad_O = Ad[:, :mesh.K_Omega]
    ud = np.linalg.solve(Ad_O, fd)

    fd_Ext = np.zeros(mesh.K)
    fd_Ext[:mesh.K_Omega] = fd
    ud_Ext = np.zeros(mesh.K)
    ud_Ext[:mesh.K_Omega] = ud

    Tstmp, fnm = filename(mesh_name, delta)
    #fileObject = open(Tstmp + fnm, 'wb')
    #pkl.dump({"Ad_O": Ad_O, "fd": fd, "ud_Ext": ud_Ext, "fd_Ext": fd_Ext, "mesh": mesh}, fileObject)
    #fileObject.close()



    plot(mesh_name, delta, Tstmp=Tstmp)

    print("Time needed", total_time)
