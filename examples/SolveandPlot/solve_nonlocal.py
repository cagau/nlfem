#!/home/klar/.venv/bin/python3
#-*- coding:utf-8 -*-

# In order to find the scripts in the python/ directory
# we add the current project path == working directory to sys-path
import sys
sys.path.append(".")
sys.path.append("../../")
import matplotlib.pyplot as plt

import numpy as np
import meshio
import pickle as pkl
from examples.SolveandPlot.conf import * # mesh_name, delta, ansatz, py_Px, py_Py, dx, dy, is_PlotSolve, boundaryConditionType
from examples.SolveandPlot.nlocal import MeshIO
from examples.SolveandPlot.aux import filename
from examples.SolveandPlot.plot import plot


try:
    from nlfem import assemble
except ImportError:
    print("\nCan't import assemble.\nTry: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libCassemble.\n")
    raise ImportError

# Necessary definitions for intersection -------------------------------------------------------------------------------
if __name__ == "__main__":
    # Nodes and Function values on nodes
    # Die Reihenfolge der phis muss zu der Reihenfolge
    # aus readmesh passen!
    # confDict = {"domainPhysicalName": "Omega", "boundaryPhysicalName": "dOmega", "interactionPhysicalName": "OmegaI"}
    confDict = {"domainPhysicalName": 1, "boundaryPhysicalName": 9, "interactiondomainPhysicalName": 2,
                "boundaryConditionType": boundaryConditionType, "ansatz": ansatz}
    # Mesh construction --------------------
    #mesh = Mesh(DATA_PATH + mesh_name + ".msh", ansatz, boundaryConditionType)
    mesh = MeshIO(DATA_PATH + mesh_name, **confDict)
    print("Delta: ", delta, "\t Mesh: ", mesh_name)
    print("Number of basis functions: ", mesh.K)

    if mesh.dim == 3:
        Ad, fd = assemble(mesh, py_Px3D, py_Py3D, dx3D, dy3D, delta, **confDict)

    if mesh.dim == 2:
        Ad, fd = assemble(mesh, py_Px, py_Py, dx, dy, delta, **confDict)

    if is_PlotSolve:
        # Solves the homogeneous Dirichlet-Problem!
        if mesh.boundaryConditionType == "Dirichlet":
            Ad_O = np.array(Ad[:, :mesh.K_Omega])
            ud = np.linalg.solve(Ad_O, fd)

            fd_Ext = np.zeros(mesh.K)
            fd_Ext[:mesh.K_Omega] = fd
            ud_Ext = np.zeros(mesh.K)
            ud_Ext[:mesh.K_Omega] = ud

    if mesh.dim == 2:
        Tstmp, fnm = filename(OUTPUT_PATH + mesh_name, delta, Tstmp=False)
        fileObject = open(Tstmp + fnm, 'wb')
        pkl.dump({"ud": ud_Ext, "fd": fd_Ext, "mesh": mesh}, fileObject)
        np.save(OUTPUT_PATH + Tstmp+"Ad_O", Ad_O)
        fileObject.close()
        mesh.point_data["ud"] = ud_Ext
        plot(OUTPUT_PATH, mesh_name, delta, Tstmp=Tstmp, maxTriangles=100)
        meshio.write(OUTPUT_PATH + mesh_name + ".vtk", mesh)
    if mesh.dim == 3:
        mesh.point_data["ud"] = ud_Ext
        meshio.write(OUTPUT_PATH + mesh_name + ".vtk", mesh)


