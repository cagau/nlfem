#-*- coding:utf-8 -*-
import numpy as np

delta = .1
ansatz = "CG"
boundaryConditionType = "Dirichlet" # "Neumann" #
model_f = "linear" # "constant" #
model_kernel = "constant" # "labeled" #
integration_method = "baryCenter" # "retriangulate" # "baryCenter" #
is_PlacePointOnCap = True
quadrule_outer = "8"
quadrule_inner = "8"

n_start = 12
n_layers = 1
N  = [n_start*2**(l) for l in list(range(n_layers))]
N_fine = N[-1]*2
u_exact = lambda x: x[0] ** 2 * x[1] + x[1] ** 2

outputdir = "results"
fnames = {
          "triPlot.pdf": outputdir+"/auto_plot.pdf",
          "tetPlot.vtk": outputdir+"/auto_plot.vtk",
          "rates.md": outputdir+"/auto_rates.md",
          "rates.pdf": outputdir+"/auto_rates.pdf",
          "timePlot.pdf": outputdir+"/timePlot.pdf",
          "report.pdf": outputdir+"/auto_report.pdf"
          }
data = {"h": [], "L2 Error": [], "Rates": [], "Assembly Time": [], "nV_Omega": []}

# Quadrature rules -----------------------------------------------------------------------------------------------------
quadrules = {
    "8" :  [
            np.array([[1.        , 0.        , 0. ],
                        [0.        , 1.        , 0. ],
                        [0.        , 0.        , 1. ],
                        [0.        , 0.        , 0. ],
                        [0.33333333, 0.33333333, 0.33333333],
                        [0.33333333, 0.33333333, 0.        ],
                        [0.33333333, 0.        , 0.33333333],
                        [0.        , 0.33333333, 0.33333333]]),
            np.array([ 0.025, 0.025, 0.025, 0.025, 0.225, 0.225, 0.225, 0.225])
            ]
}

py_Px = quadrules[quadrule_outer][0]
dx =  quadrules[quadrule_outer][1]
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

if __name__=="__main__":
    save(outputdir)
