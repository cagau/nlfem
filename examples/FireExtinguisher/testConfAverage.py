import numpy as np

def u_exact_linearRhs(x):
    return x[0] ** 2 * x[1] + x[1] ** 2
def u_exact_FieldConstantBothRhs(x):
    return np.array([x[1]**2, x[0]**2 * x[1]])*0.4

KERNELS = [
    {
        "function": "constant",
        "horizon": 0.1,
        "outputdim": 1
    }
]

LOADS = [
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
if 0:
    Px = np.array([[0.33333333, 0.33333333],
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
    dx = 0.5 * np.array([0.14431560767779,
                    0.09509163426728,
                    0.09509163426728,
                    0.09509163426728,
                    0.10321737053472,
                    0.10321737053472,
                    0.10321737053472,
                    0.03245849762320,
                    0.03245849762320,
                    0.03245849762320,
                    0.02723031417443,
                    0.02723031417443,
                    0.02723031417443,
                    0.02723031417443,
                    0.02723031417443,
                    0.02723031417443])


Py = np.array([[0.33333333, 0.33333333]])
dy = 0.5 * np.array([1.0])


CONFIGURATIONS = [
    {
        # "savePath": "pathA",
        "ansatz": "CG", #DG
        "approxBalls": {
            "method": "averageBall",
            "isPlacePointOnCap": True,  # required for "retriangulate" only
            "averageBallWeights": [1., 1., 1.]  # required for "averageBall" only
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
            "tensorGaussDegree": 5,  # Degree of tensor Gauss quadrature for weakly singular kernels.
        }
    }
    ,
    {
        # "savePath": "pathA",
        "ansatz": "CG", #DG
        "approxBalls": {
            "method": "averageBall",
            "isPlacePointOnCap": True,  # required for "retriangulate" only
            "averageBallWeights": [0., 1., 1.]  # required for "averageBall" only
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
            "tensorGaussDegree": 5,  # Degree of tensor Gauss quadrature for weakly singular kernels.
        }
    }
    ,
    {
        # "savePath": "pathA",
        "ansatz": "CG", #DG
        "approxBalls": {
            "method": "averageBall",
            "isPlacePointOnCap": True,  # required for "retriangulate" only
            "averageBallWeights": [0., 0., 1.]  # required for "averageBall" only
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
            "tensorGaussDegree": 5,  # Degree of tensor Gauss quadrature for weakly singular kernels.
        }
    }
]