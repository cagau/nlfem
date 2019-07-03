#-*- coding:utf-8 -*-

import readmesh as rm
from quadpoints import P, weights
import numpy as np
import time
# Help Functions to check whether two elements interact
from nlocal import p_in_nbhd, clsMesh, inNbhd
from aux import timestamp
# Necessary definitions for intersection -------------------------------------------------------------------------------

# Define Right side f
def fPhys(x):
    """ Right side of the equation.

    :param x: nd.array, real, shape (2,). Physical point in the 2D plane
    :return: real
    """
    # f = 1
    return 1

delta = .2
def kernelPhys(x,y):
    """ Integration kernel.

    :param x: nd.array, real, shape (2,). Physical point in the 2D plane
    :param y: nd.array, real, shape (2,). Physical point in the 2D plane
    :return: real
    """
    n = x.shape
    m = y.shape
    if not n == (2,) and m == (2,):
        raise ValueError('kernelPhys(x,y) only accepts 2D input for x and y.')
    else:
        # $\gamma(x,y) = 4 / (pi * \delta**4)$
        # Wir erwarten $u(x) = 1/4 (1 - ||x||^2)$
        return 4 / (np.pi * delta**4)


if __name__ == "__main__":
    # Nodes and Function values on nodes
    # Die Reihenfolge der phis muss zu der Reihenfolge
    # aus readmesh passen!

    psi0 = 1-P[0, :]-P[1, :]
    psi1 = P[0, :]
    psi2 = P[1, :]
    psi = np.array([psi0, psi1, psi2])

    # Allocate Matrix A and right side f
    # Mesh construction --------------------

    msh_name = "tiny"
    args = rm.read_mesh("circle_" + msh_name + ".msh")
    mesh = clsMesh(*args)
    Ad = np.zeros((mesh.K_Omega, mesh.K))
    fd = np.zeros(mesh.K_Omega)

    # Loop over triangles --------------------------------------------------------------------------------------------------
    start = time.time()
    for aTdx in range(0, mesh.J_Omega): # Laufe über 0 bis KT_Omega (der Index der Dreiecke in Omega).
        aT = mesh[aTdx]
        # Get the indices of the nodes wrt. T.E (ak) and Verts (aK) which lie in Omega.
        aBdx_O, aVdx_O = mesh.Vdx_inOmega(aTdx)

        aVdx = mesh.Vdx(aTdx) # Index for both Omega or OmegaI
        # Compute the integral and save to result to fd
        fd[aVdx_O] = (psi[aBdx_O] * fPhys(aT.toPhys(P))) @ weights * aT.absDet()

        # Set local term
        term1 = np.zeros((len(aBdx_O), 3))
        for j, abdx_O in enumerate(aBdx_O):
            for k in range(3):
                #term1[j, k] = (psi[abdx_O]*psi[k]) @ weights * aT.absDet() / delta**2 * 4
                Ad[aVdx_O[j], aVdx[k]] += (psi[abdx_O]*psi[k]) @ weights * aT.absDet() / delta**2 * 4
        #T1, T2 = np.meshgrid(aVdx_O, aVdx, indexing="ij")
        #test1 = Ad[T1, T2]
        for bTdx in range(0, mesh.J):
            bT = mesh[bTdx]
            # Term2 berechnen
            n_P = P.shape[1]
            term2 = np.zeros((3, n_P))
            X = aT.toPhys(P)
            Y = bT.toPhys(P)

            for k in range(n_P):
                 if p_in_nbhd(P[:, k], aT, bT, delta):
                    for j in range(n_P):
                        for psi_j in range(3):
                            ker = kernelPhys(Y[:, j], X[:, k])
                            product = psi[psi_j, j]*ker
                            term2[psi_j, k] += product*weights[j]*bT.absDet()

            if not (term2 == 0).all():
                #term2_ = term2[np.newaxis]
                #psi_ = psi[aBdx_O, np.newaxis, :]
                bVdx = mesh.Vdx(bTdx) #Get the indices of the nodes wrt. Verts (bK) which lie in Omega or OmegaI.
                #a = np.zeros((len(aBdx_O),3))
                for j, abdx_O in enumerate(aBdx_O):
                    for k in range(3):
                        Ad[aVdx_O[j], bVdx[k]] -= (psi[abdx_O] * term2[k]) @ weights * aT.absDet()
                #T1, T2 = np.meshgrid(aVdx_O, bVdx, indexing="ij")
                #Ad[T1, T2] += (psi_*term2_) @ weights * aT.absDet()
                #c = b-a
                #print(c)

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