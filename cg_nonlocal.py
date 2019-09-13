#!/home/klar/.venv/bin/python3
#-*- coding:utf-8 -*-
import numpy as np
from numpy.linalg import norm, solve
import pickle as pkl
import matplotlib.pyplot as plt
import time
from conf import mesh_name, delta, ansatz, py_P, weights, SOLVE
from nlocal import clsFEM#, assemble # Mesh class
from aux import filename
from plot import plot, plot_sol
from assemble import assemble, clsEvaluate, assembleMass, clsEvaluateMass
from scipy.interpolate import griddata

def pcg(Q, b, x, C, tol=1e-8, max_it=300, verbose = False):
    n = b.size
    r = Q(x) - b
    beta = .1

    p = np.zeros(n)
    h = C(r)
    res_new = r @ h

    k = 0
    log = []
    start = time.time()
    t_log = []
    res_old = 2*tol

    while np.sqrt(abs(res_old)) >= tol and k < max_it:
        k += 1
        p = -h + beta * p
        alpha = res_new/ p.dot(Q(p))
        x = x + alpha * p
        r = r + alpha * Q(p)
        h = C(r)
        res_old = res_new
        res_new = r @ h
        beta = res_new / res_old
        log.append(np.sqrt(abs(res_new)))
        t_log.append(time.time() - start)
        print("{:1.2e}".format(res_old))
    if verbose: print("Iterations: ", k)
    return {"x": x, "log": log, "time": t_log}

def cg(Q, b, x, tol=1e-8, max_it=300, verbose = False):
    n = b.size
    r = Q(x) - b
    beta = .1

    p = np.zeros(n)
    res_new = norm(r)

    k = 0
    log = []
    start = time.time()
    t_log = []
    res_old = 2 * tol

    while res_old >= tol and k < max_it:
        k += 1
        p = -r + beta * p
        alpha = res_new ** 2 / p.dot(Q(p))
        x = x + alpha * p
        r = r + alpha * Q(p)
        res_old = res_new
        res_new = norm(r)
        beta = res_new ** 2 / res_old ** 2
        log.append(res_new)
        t_log.append(time.time() - start)
    if verbose: print("Iterations: ", k)
    return {"x": x, "log": log, "time": t_log}

# Necessary definitions for intersection -------------------------------------------------------------------------------
if __name__ == "__main__":
    # Nodes and Function values on nodes
    # Die Reihenfolge der phis muss zu der Reihenfolge
    # aus readmesh passen!

    fine_mesh_name = "rectangle2"
    coarse_mesh_name = "rectangle1"
    # Mesh construction --------------------
    fine_mesh = clsFEM(fine_mesh_name + ".msh", ansatz)
    print("Delta: ", delta, "\t Mesh: ", fine_mesh_name)
    print("Number of basis functions: ", fine_mesh.K)

    coarse_mesh = clsFEM(coarse_mesh_name + ".msh", ansatz)
    print("Delta: ", delta, "\t Mesh: ", coarse_mesh_name)
    print("Number of basis functions: ", coarse_mesh.K)

    # Initialize Evaluator Object
    A = clsEvaluate(fine_mesh.K, fine_mesh.K_Omega, fine_mesh.J, fine_mesh.J_Omega, fine_mesh.L, fine_mesh.L_Omega, fine_mesh.T, fine_mesh.V, py_P, weights, weights, delta)
    ud = np.zeros(fine_mesh.K_Omega)
    fd = A.get_f()

    # Initialize Coarse Evaluator Object
    coarse_A = clsEvaluate(coarse_mesh.K, coarse_mesh.K_Omega, coarse_mesh.J, coarse_mesh.J_Omega, coarse_mesh.L, coarse_mesh.L_Omega, coarse_mesh.T, coarse_mesh.V, py_P, weights, weights, delta)

    # Precoditioner
    class clsMultilevel:
        def __init__(self, fine_points, coarse_points, coarse_A):
            self.coarse_points = coarse_points
            self.fine_points = fine_points
            self.coarse_A = coarse_A

        def project(self, fine_u):
            return griddata(self.fine_points, fine_u, self.coarse_points, method="nearest")

        def interpolate(self, coarse_u):
            return griddata(self.coarse_points, coarse_u, self.fine_points, method="linear", fill_value=0)
        def __call__(self, fine_u):
            coarse_u = self.project(fine_u)
            coarse_sol = cg(self.coarse_A, coarse_u, coarse_u, max_it=30, verbose=True)["x"]
            sol = self.interpolate(coarse_sol)
            return sol

    M = clsMultilevel(fine_mesh.V[:fine_mesh.K_Omega], coarse_mesh.V[:coarse_mesh.K_Omega], coarse_A)

    x = np.zeros(fine_mesh.K_Omega)

    sol_pcg = pcg(A, fd, x, M, verbose=True)
    sol_cg = cg(A, fd, x, verbose=True)
    ud_Ext = np.zeros(fine_mesh.K)
    ud_Ext[:fine_mesh.K_Omega] = sol_pcg["x"]

    plot_sol(fine_mesh, ud_Ext, delta, Tstmp= "", mesh_name="plot_solution", sol_logs={"CG": sol_cg, "PCG": sol_pcg})



