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

    msh_name = "medium"
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

        for bTdx in range(0, mesh.J):
            bT = mesh[bTdx]
            n_P = P.shape[1]
            term2 = np.zeros((3, n_P))
            X = aT.toPhys(P)
            Y = bT.toPhys(P)

            if inNbhd(aT, bT, delta, norm="l2"):
                bVdx = mesh.Vdx(bTdx)
                for i, avdx in enumerate(aVdx_O):
                    a = aBdx_O[i]
                    for b in range(3):
                        g = kernelPhys(np.zeros(1), np.zeros(1))
                        Ad[avdx, bVdx[b]] -= aT.absDet()*bT.absDet() * (psi[a] @ weights) * (psi[b] @ weights * g)
                        Ad[avdx, aVdx[b]] += aT.absDet()*bT.absDet() * (psi[a] * psi[b]) @ weights * g
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