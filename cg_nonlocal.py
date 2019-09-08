#!/home/klar/.venv/bin/python3
#-*- coding:utf-8 -*-
import numpy as np
import pickle as pkl

from conf import mesh_name, delta, ansatz, py_P, weights, SOLVE
from nlocal import clsFEM#, assemble # Mesh class
from aux import filename
from plot import plot
from assemble import assemble, evaluate

# Necessary definitions for intersection -------------------------------------------------------------------------------
if __name__ == "__main__":
    # Nodes and Function values on nodes
    # Die Reihenfolge der phis muss zu der Reihenfolge
    # aus readmesh passen!

    # Mesh construction --------------------
    mesh = clsFEM("circle_" + mesh_name + ".msh", ansatz)
    print("Delta: ", delta, "\t Mesh: ", mesh_name)
    print("Number of basis functions: ", mesh.K)
    ud = np.array([1.] * mesh.K_Omega + [0.] * (mesh.K - mesh.K_Omega))
    ud[:mesh.K_Omega] = np.random.rand(mesh.K_Omega)
    Ad, fd  = assemble(mesh.K, mesh.K_Omega, mesh.J, mesh.J_Omega, mesh.L, mesh.L_Omega, mesh.T, mesh.V, py_P, weights, weights, delta)
    vd, fd_ev  = evaluate(ud, mesh.K, mesh.K_Omega, mesh.J, mesh.J_Omega, mesh.L, mesh.L_Omega, mesh.T, mesh.V, py_P, weights, weights, delta)

    Ad_O = np.array(Ad[:, :mesh.K_Omega])
    print("\nEvaluate :\n", vd)
    print("\nAssemble :\n", Ad_O @ ud[:mesh.K_Omega])
    print("\nTest :\n", np.linalg.norm(Ad_O @ ud[:mesh.K_Omega] - vd))