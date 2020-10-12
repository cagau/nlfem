#-*- coding:utf-8 -*-
import numpy as np

delta = .1
ansatz = "CG"
boundaryConditionType = "Dirichlet" # "Neumann" #
model_f = "constant" # "constant" #
model_kernel = "constant1D"#"parabola" "linearPrototypeMicroelastic"# "constant" # "labeled" #
integration_method = "baryCenter" # "tensorgauss" "retriangulate" # "baryCenter" # subSetBall # superSetBall averageBall
is_PlacePointOnCap = True
quadrule_outer = 6
quadrule_inner = 1
tensorGaussDegree = 4

a, b = -1, 1
n_start = 12
n_layers = 5
N  = [n_start*2**(l) for l in list(range(n_layers))]
N_fine = N[-1]*4
def u_exact(x):
    return -x ** 2/2.

fnames = {"triPlot.pdf": "results/auto_plot.pdf",
          "rates.md": "results/auto_rates.md",
          "rates.pdf": "results/auto_rates.pdf",
          "timePlot.pdf": "results/timePlot.pdf",
          "report.pdf": "results/auto_report.pdf"}
data = {"h": [], "L2 Error": [], "Rates": [], "Assembly Time": [], "nV_Omega": []}

# Quadrature rules -----------------------------------------------------------------------------------------------------
py_Px, dx = np.polynomial.legendre.leggauss(quadrule_outer)
py_Px += 1
py_Px /= 2.
dx /= 2.

py_Py, dy = np.polynomial.legendre.leggauss(quadrule_inner)
py_Py += 1
py_Py /= 2.
dy /= 2.

def writeattr(file, attr_name):
    file.write(attr_name+"\n")
    file.write(str(eval(attr_name))+"\n")

def save(path):
    # Save Configuration
    confList = [
        "model_kernel",
        "model_f",
        "integration_method",
        "is_PlacePointOnCap"]

    f = open(path + "/conf", "w+")
    [writeattr(f, attr_name) for attr_name in confList]
    f.close()

    # Provide Quadrature Rules
    py_Px.tofile(path+"/quad.Px")
    py_Py.tofile(path+"/quad.Py")
    dx.tofile((path+"/quad.dx"))
    dy.tofile((path+"/quad.dy"))
