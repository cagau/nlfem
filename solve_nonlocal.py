#-*- coding:utf-8 -*-
import numpy as np
import pickle as pkl

from conf import mesh_name, delta, ansatz
from nlocal import clsFEM # Mesh class
from aux import filename
from plot import plot
import assemble

# Necessary definitions for intersection -------------------------------------------------------------------------------
if __name__ == "__main__":
    # Nodes and Function values on nodes
    # Die Reihenfolge der phis muss zu der Reihenfolge
    # aus readmesh passen!

    # Mesh construction --------------------
    mesh = clsFEM("circle_" + mesh_name + ".msh", ansatz)
    print("Delta: ", delta, "\t Mesh: ", mesh_name)
    print("Number of basis functions: ", mesh.K)

    Ad, fd  = assemble.assemble(mesh)
    Ad_O = Ad[:, :mesh.K_Omega]
    ud = np.zeros(mesh.K_Omega)
    #ud = np.linalg.solve(Ad_O, fd)

    fd_Ext = np.zeros(mesh.K)
    fd_Ext[:mesh.K_Omega] = fd
    ud_Ext = np.zeros(mesh.K)
    ud_Ext[:mesh.K_Omega] = ud

    Tstmp, fnm = filename(mesh_name, delta, Tstmp=False)
    fileObject = open(Tstmp + fnm, 'wb')
    pkl.dump({"Ad_O": Ad_O, "fd": fd, "ud_Ext": ud_Ext, "fd_Ext": fd_Ext, "mesh": mesh}, fileObject)
    fileObject.close()

    plot(mesh_name, delta, Tstmp=Tstmp)


