#!/home/klar/.venv/bin/python3
#-*- coding:utf-8 -*-
import numpy as np
import pickle as pkl

from conf import mesh_name, delta, ansatz, py_P, weights
from nlocal import clsFEM#, assemble # Mesh class
from aux import filename
from plot import plot
from assemble import assemble

# Necessary definitions for intersection -------------------------------------------------------------------------------
if __name__ == "__main__":
    # Nodes and Function values on nodes
    # Die Reihenfolge der phis muss zu der Reihenfolge
    # aus readmesh passen!

    # Mesh construction --------------------
    mesh = clsFEM("circle_" + mesh_name + ".msh", ansatz)
    print("Delta: ", delta, "\t Mesh: ", mesh_name)
    print("Number of basis functions: ", mesh.K)

    Ad, fd  = assemble(mesh.K, mesh.K_Omega, mesh.J, mesh.J_Omega, mesh.L, mesh.L_Omega,
                       mesh.T,
                       mesh.V,
                       py_P, weights, weights, delta)
"""    Ad_O = Ad[:, :mesh.K_Omega]
    ud = np.linalg.solve(Ad_O, fd)

    fd_Ext = np.zeros(mesh.K)
    fd_Ext[:mesh.K_Omega] = fd
    ud_Ext = np.zeros(mesh.K)
    ud_Ext[:mesh.K_Omega] = ud

    Tstmp, fnm = filename(mesh_name, delta, Tstmp=False)
    fileObject = open(Tstmp + fnm, 'wb')
    pkl.dump({"Ad_O": Ad_O, "fd": fd, "ud_Ext": ud_Ext, "fd_Ext": fd_Ext, "mesh": mesh}, fileObject)
    np.save(Tstmp+"Ad_O", Ad_O)
    fileObject.close()

    plot(mesh_name, delta, Tstmp=Tstmp)"""


