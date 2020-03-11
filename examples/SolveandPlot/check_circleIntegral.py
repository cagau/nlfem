#!/home/klar/.venv/bin/python3
#-*- coding:utf-8 -*-

# In order to find the scripts in the python/ directory
# we add the current project path == working directory to sys-path
import sys
sys.path.append(".")

import numpy as np
import pickle as pkl
from python.conf import *
from python.nlocal import Mesh#, assemble # Mesh class
from python.aux import filename
from python.plot import plot

from assemble import py_check_par_assemble
from matplotlib.backends.backend_pdf import PdfPages

# Necessary definitions for intersection -------------------------------------------------------------------------------
if __name__ == "__main__":
    # Nodes and Function values on nodes
    # Die Reihenfolge der phis muss zu der Reihenfolge
    # aus readmesh passen!

    # Mesh construction --------------------
    #mesh = Mesh(mesh_name + ".msh", ansatz)
    #print("Delta: ", delta, "\t Mesh: ", mesh_name)
    #mesh.triangles = np.load("../compare_data/Triangles_Chris.npy")
    #mesh.vertices = np.load("../compare_data/Verts_Chris.npy")

    import sys
    #sys.path.insert(1, '../nonlocal-assembly-chris')
    #mesh = Mesh(pkl.load(open( "../compare_data/mesh.pkl", "rb" )), ansatz)

    confDict = {"domainPhysicalName": 1, "boundaryPhysicalName": 9, "interactiondomainPhysicalName": 2,
                "boundaryConditionType": boundaryConditionType, "ansatz": ansatz}
    # Mesh construction --------------------
    #mesh = Mesh(DATA_PATH + mesh_name + ".msh", ansatz, boundaryConditionType)
    mesh = Mesh(DATA_PATH + mesh_name, **confDict)

    print("Number of basis functions: ", mesh.K)
    pp = PdfPages(OUTPUT_PATH + "check_circlIntegral" + ".pdf")
    fd = py_check_par_assemble(mesh, py_Px, dx, delta, pp  =pp)
    #check = fd-delta**2*np.pi
    #Neighbours = np.load("check_circleIntegral.npy")
    pp.close()
    print("Stop")


