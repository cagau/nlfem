import numpy as np

def u_exact_linearRhs(x):
    return x[0] ** 2 * x[1] + x[1] ** 2
def u_exact_FieldConstantBothRhs(x):
    return np.array([x[1]**2, x[0]**2 * x[1]])*0.4

KERNELS = [
    {
       "function": "linearPrototypeMicroelastic",
       "horizon": 0.3,# Due to the very simplistic mesh generation we are limited to delta D/10., where D in N.
       "outputdim": 1,
        "fractional_s": -0.5
    },
    {
        "function": "fractional",
        "horizon": 0.3,# Due to the very simplistic mesh generation we are limited to delta D/10., where D in N.
        "outputdim": 1,
        "fractional_s": 0.5
    }
]

LOADS = [
    {"function": "linear", "solution": u_exact_linearRhs},
    {"function": "linear", "solution": u_exact_linearRhs}
]


Px = np.array([[0.33333333333333,    0.33333333333333],
                  [0.47014206410511,    0.47014206410511],
                  [0.47014206410511,    0.05971587178977],
                  [0.05971587178977,    0.47014206410511],
                  [0.10128650732346,    0.10128650732346],
                  [0.10128650732346,    0.79742698535309],
                  [0.79742698535309,    0.10128650732346]])
dx = 0.5 * np.array([0.22500000000000,
                        0.13239415278851,
                        0.13239415278851,
                        0.13239415278851,
                        0.12593918054483,
                        0.12593918054483,
                        0.12593918054483])

Py = Px
dy = dx

CONFIGURATIONS = [
    {
        # "savePath": "pathA",
        "ansatz": "CG", #DG
        "approxBalls": {
            "method": "fractional",#"fractional",
            "isPlacePointOnCap": True,  # required for "retriangulate" only
            #"averageBallWeights": [1., 1., 1.]  # required for "averageBall" only
        },
        "quadrature": {
            "outer": {
                "points": Px,
                "weights": dx
            },
            "inner": {
                "points": Py,
                "weights": dy
            },
            "tensorGaussDegree": 4  # Degree of tensor Gauss quadrature for weakly singular kernels.
        }
    }
]
