import numpy as np
import quadpy

def u_exact_linearRhs(x):
    return x[0] ** 2 * x[1] + x[1] ** 2
def u_exact_FieldConstantBothRhs(x):
    return np.array([x[1]**2, x[0]**2 * x[1]])*0.4

KERNELS = [
    {
        "function": "fractional",
        "horizon": 0.5,# Due to the very simplistic mesh generation we are limited to delta D/10., where D in N.
        "outputdim": 1,
        "fractional_s": 0.9
    }
]

LOADS = [
    {"function": "linear", "solution": u_exact_linearRhs}
]

quadrules = {
    "7point": [
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
    "dunavant":
    [
        quadpy.triangle.dunavant_20().points[:, :2].copy(),
        quadpy.triangle.dunavant_20().weights*0.5
    ],
    "vireanu":
    [
        quadpy.triangle.vioreanu_rokhlin_19().points[:, :2].copy(),
        quadpy.triangle.vioreanu_rokhlin_19().weights*0.5
    ]
}

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
                "points": quadrules["7point"][0],
                "weights": quadrules["7point"][1]
            },
            "inner": {
                "points": quadrules["7point"][0],
                "weights": quadrules["7point"][1]
            },
            "tensorGaussDegree": 5 # Degree of tensor Gauss quadrature for weakly singular kernels.
        }
    }
]
