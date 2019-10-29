#!/home/klar/.venv/bin/python3
#-*- coding:utf-8 -*-
import numpy as np
import time
from conf import mesh_name, delta, ansatz, py_P, weights, is_PlotSolve
from nlocal import Mesh
from assemble import assemble, clsEvaluate

# Necessary definitions for intersection -------------------------------------------------------------------------------
if __name__ == "__main__":
    # Nodes and Function values on nodes
    # Die Reihenfolge der phis muss zu der Reihenfolge
    # aus readmesh passen!

    #mesh_name = "circle_large"
    # Mesh construction --------------------
    mesh = Mesh(mesh_name + ".msh", ansatz)
    print("Delta: ", delta, "\t Mesh: ", mesh_name)
    print("Number of basis functions: ", mesh.K)

    # Initialize Evaluator Object
    A = clsEvaluate(mesh.K, mesh.K_Omega, mesh.nE, mesh.nE_Omega, mesh.nV, mesh.nV_Omega, mesh.triangles, mesh.vertices, py_P, weights, weights, delta)
    Ad, fd = assemble(mesh.K, mesh.K_Omega, mesh.nE, mesh.nE_Omega, mesh.nV, mesh.nV_Omega,
                      mesh.triangles,
                      mesh.vertices,
                      py_P, weights, weights, delta)
    ud = np.ones(mesh.K_Omega)
    testvd = Ad[:, :mesh.K_Omega] @ ud
    fd = A.get_f()
    start = time.time()
    
    vd = A(ud)
    total_time = time.time() - start
    print("Evaluation Time\t", "{:1.2e}".format(total_time), " Sec")
    print("Error: ", np.linalg.norm( vd- testvd))
