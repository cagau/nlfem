import numpy as np
from mesh import read_gmsh, regular_square, regular_cube
from quadrules import quadrules


def zeros(x):
    return 0.0


def linear(x):
    return x[0] ** 2 * x[1] + x[1] ** 2


def field(x):
    return np.array([x[1]**2, x[0]**2 * x[1]])*0.4


cfg_dict = {
    "disjoint_av": {
        "kernel":
            {
                "function": "labeledValve",
                "horizon": 0.1,
                "outputdim": 1
            },
        "load":
            {
                "function": "constant",
                "dirichlet": zeros
            },
        "conf":
            {
                "ansatz": "CG",
                "approxBalls": {
                    "method": "retriangulate"
                },
                "quadrature": {
                    "outer": {
                        "points": quadrules["2d7p"]["points"],
                        "weights": quadrules["2d7p"]["weights"]
                    },
                    "inner": {
                        "points": quadrules["2d7p"]["points"],
                        "weights": quadrules["2d7p"]["weights"]
                    },
                },
                "verbose": False
            },
        "mesh":  read_gmsh("data/disjoint.msh")
    },
    "hole_noav": {
        "kernel":
            {
                "function": "constant",
                "horizon": 0.1,
                "outputdim": 1
            },
        "load":
            {
                "function": "constant",
                "dirichlet": zeros
            },
        "conf":
            {
                "ansatz": "CG", #DG
                "approxBalls": {
                    "method": "retriangulate"
                },
                "quadrature": {
                    "outer": {
                        "points": quadrules["2d7p"]["points"],
                        "weights": quadrules["2d7p"]["weights"]
                    },
                    "inner": {
                        "points": quadrules["2d7p"]["points"],
                        "weights": quadrules["2d7p"]["weights"]
                    },
                },
                "verbose": False
            },
        "mesh":  read_gmsh("data/hole.msh", set_art_vertex=False)
    },
    "hole_av": {
        "kernel":
            {
                "function": "constant",
                "horizon": 0.1,
                "outputdim": 1
            },
        "load":
            {
                "function": "constant",
                "dirichlet": zeros
            },
        "conf":
            {
                "ansatz": "CG", #DG
                "approxBalls": {
                    "method": "retriangulate"
                },
                "quadrature": {
                    "outer": {
                        "points": quadrules["2d7p"]["points"],
                        "weights": quadrules["2d7p"]["weights"]
                    },
                    "inner": {
                        "points": quadrules["2d7p"]["points"],
                        "weights": quadrules["2d7p"]["weights"]
                    },
                },
                "verbose": False
            },
        "mesh":  read_gmsh("data/hole.msh")
    },
    "peridynamics": {
        "kernel":
            {
                "function": "linearPrototypeMicroelasticField",
                "horizon": 0.1,
                "outputdim": 2,
                "fractional_s": -0.5
            },
        "load":
            {
                "function": "linearField",
                "dirichlet": field
            },
        "conf":
            {
                "ansatz": "CG", #DG
                "approxBalls": {
                    "method": "retriangulate"
                },
                "closeElements": "retriangulate", #weakSingular
                "quadrature": {
                    "outer": {
                        "points": quadrules["2d16p"]["points"],
                        "weights": quadrules["2d16p"]["weights"]
                    },
                    "inner": {
                        "points": quadrules["2d7p"]["points"],
                        "weights": quadrules["2d7p"]["weights"]
                    },
                    "tensorGaussDegree": 1,
                },
                "verbose": False
            },
        "mesh":  read_gmsh("data/circle.msh")
    },
}