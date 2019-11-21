#!/home/klar/.venv/bin/python3
#-*- coding:utf-8 -*-

# In order to find the scripts in the python/ directory
# we add the current project path == working directory to sys-path
import sys
sys.path.append(".")

import numpy as np
import pickle as pkl
from python.conf import * # mesh_name, delta, ansatz, py_Px, py_Py, dx, dy, is_PlotSolve, boundaryConditionType
from python.nlocal import Mesh
from python.aux import filename
from python.plot import plot
try:
    from assemble import assemble
except ImportError:
    print("\nCan't import assemble.\nTry: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libCassemble.\n")
    raise ImportError

# Necessary definitions for intersection -------------------------------------------------------------------------------
if __name__ == "__main__":
    # Nodes and Function values on nodes
    # Die Reihenfolge der phis muss zu der Reihenfolge
    # aus readmesh passen!

    # Mesh construction --------------------
    mesh = Mesh(DATA_PATH + mesh_name + ".msh", ansatz, boundaryConditionType)
    print("Delta: ", delta, "\t Mesh: ", mesh_name)
    print("Number of basis functions: ", mesh.K)

    Ad, fd = assemble(mesh, py_Px, py_Py, dx, dy, delta)

    if is_PlotSolve:
        # Solves the homogeneous Dirichlet-Problem!
        Ad_O = np.array(Ad[:,:mesh.K_Omega])
        ud = np.linalg.solve(Ad_O, fd)

        fd_Ext = np.zeros(mesh.K)
        fd_Ext[:mesh.K_Omega] = fd
        ud_Ext = np.zeros(mesh.K)
        ud_Ext[:mesh.K_Omega] = ud

        Tstmp, fnm = filename(OUTPUT_PATH + mesh_name, delta, Tstmp=False)
        fileObject = open(Tstmp + fnm, 'wb')
        pkl.dump({"ud": ud_Ext, "fd": fd_Ext, "mesh": mesh}, fileObject)
        np.save(OUTPUT_PATH + Tstmp+"Ad_O", Ad_O)
        fileObject.close()

        plot(OUTPUT_PATH, mesh_name, delta, Tstmp=Tstmp, maxTriangles=100)

