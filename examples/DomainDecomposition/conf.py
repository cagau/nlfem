#-*- coding:utf-8 -*-
import numpy as np
from examples.DomainDecomposition.nlocal import timestamp

DATA_PATH = "data/"
OUTPUT_PATH = "output/"
isOverride = False
tmpstp = timestamp()
geofile = "DD_nonfloating" # .geo file
mesh_name = geofile

element_size = 0.05 # to control grid size via gmsh (element size factor)
delta = 0.1 # interaction horizon (attention: here only l2-norm)

ansatz = "CG"
boundaryConditionType = "Dirichlet" # "Neumann" #
model_f = "linear" # "constant" #
model_kernel = "constant" # "labeled" #
integration_method = "retriangulate" # "retriangulate" # "baryCenter" #
is_PlacePointOnCap = False
quadrule_outer = "16"
quadrule_inner = "1"

def eval_g(x):
    return 0
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