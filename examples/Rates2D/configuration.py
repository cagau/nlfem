import numpy as np

mesh = {
    # "maxDiameter": 0.0
    "vertices": np.load("data/vertices.npy"),
    "vertexLabels": np.load("data/verticesLabels.npy"),
    "elements": np.load("data/elements.npy"),
    "elementLabels": np.load("data/elementsLabels.npy")
}

kernel = {
            "function": "constant",
            "horizon": 0.1
}

load = {
        "function": "linear"
}

configuration = {
    # "savePath": "pathA",
    "ansatz": "CG", #DG
    "approxBalls": "baryCenter", #baryCenter, retriangulate, baryCenterRT, averageBall
    "quadrature": {
        "outer": {
            "points": np.load("data/Px.npy"),
            "weights": np.load("data/dx.npy")
        },
        "inner": {
            "points": np.load("data/Py.npy"),
            "weights": np.load("data/dy.npy")
        },
        # "tensorGaussDegree": 4, # Degree of tensor Gauss quadrature for weakly singular kernels.
    }
}
