#!/home/klar/.venv/bin/python3
#-*- coding:utf-8 -*-
import numpy as np
import pickle as pkl
import time
from conf import mesh_name, delta, ansatz, py_P, weights, SOLVE
from nlocal import clsFEM#, assemble # Mesh class
from aux import filename
from plot import plot_convRate, plot_sol
from assemble import assemble, clsEvaluateMass
from scipy.interpolate import griddata

def solveNonlocal(mesh_name):
    # Mesh construction --------------------
    mesh = clsFEM(mesh_name+".msh", ansatz)
    print("\nDelta: ", delta, "\t Mesh: ", mesh_name)
    print("Number of basis functions: ", mesh.K)

    Ad, fd = assemble(mesh.K, mesh.K_Omega, mesh.J, mesh.J_Omega, mesh.L, mesh.L_Omega,
                       mesh.T,
                       mesh.V,
                       py_P, weights, weights, delta)

    Ad_O = np.array(Ad[:, :mesh.K_Omega])

    start = time.time()
    ud = np.linalg.solve(Ad_O, fd)
    ud_Ext = np.zeros(mesh.K)
    ud_Ext[:mesh.K_Omega] = ud

    print("Solve Time\t", "{:1.2e}".format(time.time() - start), "Sec")
    return ud_Ext, mesh

def uLocaSol(x,y):
    n = x.shape[0]
    u =  1/4*(1 - x**2 - y**2)
    u = np.maximum(u, np.zeros(n))
    return u

# Necessary definitions for intersection -------------------------------------------------------------------------------
if __name__ == "__main__":
    data = {"name": [], "h": [], "error": [], "log2 relerror": [], "Time": []}

    start_total = time.time()
    mesh_names = ["rectangle0", "rectangle1", "rectangle2", "rectangle3", "rectangle4"]
    data["name"] = mesh_names
    data["h"] = [.1, .05, .025, .0125, .00625]

    data["log2 relerror"] = []

    u_base, baseMesh = solveNonlocal(mesh_name=data["name"][-1])
    #plot_sol(baseMesh, u_base, delta=delta, Tstmp="", mesh_name=data["name"][-1])
    u_base = u_base[:baseMesh.K_Omega]

    M = clsEvaluateMass(baseMesh.K_Omega, baseMesh.J_Omega, baseMesh.T, baseMesh.V, py_P, weights)

    for mesh_name in data["name"][:-1]:
        u, mesh = solveNonlocal(mesh_name=mesh_name)
        #plot_sol(mesh, u, delta=delta, Tstmp= "",  mesh_name=mesh_name)
        start = time.time()
        u_inp = griddata(mesh.V, u, baseMesh.V, method="linear")[:baseMesh.K_Omega]
        print("Interpol Time\t", "{:1.2e}".format(time.time() - start), "Sec")

        L2err = (u_inp - u_base) @ M(u_inp - u_base)
        data["error"].append(L2err)
        print("L2 Distance\t", L2err)

    for k in range(len(mesh_names)-2):
        logErr = np.log2(data["error"][k]/data["error"][k+1])
        data["log2 relerror"].append(logErr)


    print("\nTotal Time\t", time.time() - start_total)
    plot_convRate(data, delta, file_name="convergence_rate", Tstmp= "")

