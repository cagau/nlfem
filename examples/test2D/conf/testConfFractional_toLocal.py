import numpy as np

def u_exact_tensorsin(x):
    return np.sin(x[0]*4*np.pi)*np.sin(x[1]*4*np.pi)

KERNELS = [
    {
        "function": "fractional",
        "horizon": None,
        "fractional_s": 0.5,
        "outputdim": 1
    }
]

LOADS = [
    {"function": "tensorsin", "solution": u_exact_tensorsin}
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

Py = np.array([[0.33333333, 0.33333333]])
dy = 0.5 * np.array([1.0])

CONFIGURATIONS = [
    {
        # "savePath": "pathA",
        "ansatz": "CG", #DG
        "is_fullConnectedComponentSearch": 0,
        "approxBalls": {
            "method": "retriangulate",
            "isPlacePointOnCap": True,  # required for "retriangulate" only
            #"averageBallWeights": [1., 1., 1.]  # required for "averageBall" only
        },
        "closeElements": "fractional",
        "quadrature": {
            "outer": {
                "points": Px,
                "weights": dx
            },
            "inner": {
                "points": Px,
                "weights": dx
            },
            "tensorGaussDegree": 4,  # Degree of tensor Gauss quadrature for weakly singular kernels.
        },
        "verbose": True
    }
]
