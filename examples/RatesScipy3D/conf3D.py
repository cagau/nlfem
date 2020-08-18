#-*- coding:utf-8 -*-
import numpy as np
import quadpy
import datetime
import os

delta = .1
ansatz = "CG"
boundaryConditionType = "Dirichlet" # "Neumann" #
model_f = "linear3D" # "constant" #
model_kernel = "constant3D" # "labeled" #
integration_method = "baryCenter" # "retriangulate" # "baryCenter" #
is_PlacePointOnCap = True
quadrule_outer = "keast_9"
quadrule_inner = "keast_4"

n_start = 12
n_layers = 2
N  = [n_start*2**(l) for l in list(range(n_layers))]
N_fine = N[-1]*2
u_exact = lambda x: x[0]**2 * x[1] + x[1]**2 + x[2]**2


outputdir = datetime.datetime.now().strftime("%m.%d_%H-%M-%S")+"_results"
os.mkdir(outputdir)
#outputdir = "results"

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
    "keast_1":
        quadpy.tetrahedron.keast_1(),
     "keast_4":
        quadpy.tetrahedron.keast_4(),
    "keast_8":
        quadpy.tetrahedron.keast_8(),
    "keast_9":
        quadpy.tetrahedron.keast_9(),
    #"stroud_1961":
    #    quadpy.nsimplex.stroud_1961(3),
    #"hammer_stroud_1a":
    #    quadpy.nsimplex.hammer_stroud_1a(3),
    #"walkington_7":
    #    quadpy.nsimplex.walkington_7(3)
}
#
#py_Px = quadrules[quadrule_outer].points[1:,:3]
#dx =  quadrules[quadrule_outer].weights/6
#py_Py = quadrules[quadrule_inner].points[1:,:3]
#dy = quadrules[quadrule_inner].weights/6

#scheme = quadpy.nsimplex.walkington_7(3)
py_Px = quadrules[quadrule_outer].points[:, :3]
dx =  quadrules[quadrule_outer].weights/6.

py_Py = quadrules[quadrule_inner].points[:, :3]
dy = quadrules[quadrule_inner].weights/6.

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
