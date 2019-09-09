#!/home/klar/.venv/bin/python3
#-*- coding:utf-8 -*-
import numpy as np
from numpy.linalg import norm
import pickle as pkl
import matplotlib.pyplot as plt

from conf import mesh_name, delta, ansatz, py_P, weights, SOLVE
from nlocal import clsFEM#, assemble # Mesh class
from aux import filename
from plot import plot
from assemble import assemble, clsEvaluate

def cg(Q, b, x, tol=1e-12, max_it=100):
    n = b.size
    beta = 0
    p = np.zeros(n)
    r = Q(x) + b
    res_new = norm(r)
    k = 0
    log = []

    while res_new >= tol and k < max_it:
        print(k, end="\r", flush=True)
        k += 1
        p = -r + beta * p
        alpha = res_new ** 2 / p.dot(Q(p))
        x = x + alpha * p
        r = r + alpha * Q(p)
        res_old = res_new
        res_new = norm(r)
        beta = res_new ** 2 / res_old ** 2
        log.append(res_new)
    print("Iterations: ", k)
    return {"x": x, "log": log}

# Necessary definitions for intersection -------------------------------------------------------------------------------
if __name__ == "__main__":
    # Nodes and Function values on nodes
    # Die Reihenfolge der phis muss zu der Reihenfolge
    # aus readmesh passen!

    # Mesh construction --------------------
    mesh = clsFEM("circle_" + mesh_name + ".msh", ansatz)
    print("Delta: ", delta, "\t Mesh: ", mesh_name)
    print("Number of basis functions: ", mesh.K)

    # Assemble full system as a Test
    Ad_test, fd_test  = assemble(mesh.K, mesh.K_Omega, mesh.J, mesh.J_Omega, mesh.L, mesh.L_Omega, mesh.T, mesh.V, py_P, weights, weights, delta)
    Ad_O = np.array(Ad_test[:, :mesh.K_Omega])

    # Initialize Evaluator Object
    A = clsEvaluate(mesh.K, mesh.K_Omega, mesh.J, mesh.J_Omega, mesh.L, mesh.L_Omega, mesh.T, mesh.V, py_P, weights, weights, delta)
    ud = np.zeros(mesh.K_Omega)

    fd = A.get_f()
    vd = A(ud)
    x = np.zeros(mesh.K_Omega)
    sol = cg(A, -fd, x)
    sol_test  = np.linalg.solve(Ad_O, fd_test)
    print("\nTest Solve:\n", np.linalg.norm(sol_test- sol["x"]))
    print("Test Eval:\n", np.linalg.norm(Ad_O @ ud - A(ud)))
    print("Test RS:\n", np.linalg.norm(fd_test - A.get_f()))

    plt.subplot(211)
    plt.plot(np.log10(sol["log"]))
    plt.title( "CG-Iteration Residuals")

    #plt.subplot(212)
    #plt.plot(np.linalg.eig(Ad_O)[0], '.')
    #plt.title("Eigenvalues")

    plt.show()





