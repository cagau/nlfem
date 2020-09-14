#-*- coding:utf-8 -*-
import numpy as np

delta = .1
ansatz = "CG"
boundaryConditionType = "Dirichlet" # "Neumann" #
model_f = "linear" # "constant" #
model_kernel = "constant" # "labeled" #
integration_method = "retriangulate" # "retriangulate" # "baryCenter" # subSetBall # superSetBall averageBall
is_PlacePointOnCap = True
quadrule_outer = "16"
quadrule_inner = "1"

n_start = 24
n_layers = 4
N  = [n_start*2**(l) for l in list(range(n_layers))]
N_fine = N[-1]*4
def u_exact(x):
    return x[0] ** 2 * x[1] + x[1] ** 2

fnames = {"triPlot.pdf": "results/auto_plot.pdf",
          "rates.md": "results/auto_rates.md",
          "rates.pdf": "results/auto_rates.pdf",
          "timePlot.pdf": "results/timePlot.pdf",
          "report.pdf": "results/auto_report.pdf"}
data = {"h": [], "L2 Error": [], "Rates": [], "Assembly Time": [], "nV_Omega": []}

# Quadrature rules -----------------------------------------------------------------------------------------------------
quadrules = {
    "7":    [
        np.array([[0.33333333333333,    0.33333333333333],
                  [0.47014206410511,    0.47014206410511],
                  [0.47014206410511,    0.05971587178977],
                  [0.05971587178977,    0.47014206410511],
                  [0.10128650732346,    0.10128650732346],
                  [0.10128650732346,    0.79742698535309],
                  [0.79742698535309,    0.10128650732346]]),

        0.5 * np.array([0.22500000000000,
                        0.13239415278851,
                        0.13239415278851,
                        0.13239415278851,
                        0.12593918054483,
                        0.12593918054483,
                        0.12593918054483])
    ],
    "16":    [
        np.array([[0.33333333, 0.33333333],
                  [0.45929259, 0.45929259],
                  [0.45929259, 0.08141482],
                  [0.08141482, 0.45929259],
                  [0.17056931, 0.17056931],
                  [0.17056931, 0.65886138],
                  [0.65886138, 0.17056931],
                  [0.05054723, 0.05054723],
                  [0.05054723, 0.89890554],
                  [0.89890554, 0.05054723],
                  [0.26311283, 0.72849239],
                  [0.72849239, 0.00839478],
                  [0.00839478, 0.26311283],
                  [0.72849239, 0.26311283],
                  [0.26311283, 0.00839478],
                  [0.00839478, 0.72849239]]),

        0.5 * np.array([0.14431560767779
                           , 0.09509163426728
                           , 0.09509163426728
                           , 0.09509163426728
                           , 0.10321737053472
                           , 0.10321737053472
                           , 0.10321737053472
                           , 0.03245849762320
                           , 0.03245849762320
                           , 0.03245849762320
                           , 0.02723031417443
                           , 0.02723031417443
                           , 0.02723031417443
                           , 0.02723031417443
                           , 0.02723031417443
                           , 0.02723031417443])
    ],
    "1":    [
        np.array([[0.33333333, 0.33333333]]),
        0.5 * np.array([1.0])
    ]
}

py_Px = quadrules[quadrule_outer][0]
dx = quadrules[quadrule_outer][1]
py_Py = quadrules[quadrule_inner][0]
dy = quadrules[quadrule_inner][1]

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