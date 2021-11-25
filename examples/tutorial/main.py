import numpy as np
from scipy.sparse.linalg import spsolve
from nlfem import loadVector, stiffnessMatrix_fromArray
from mesh import trim_artificial_vertex
from plot import saveplot, showplot
from conf import cfg_dict
import os

if __name__ == "__main__":
    if not os.path.isdir("plots"):
        os.mkdir("plots")

    for cfg_name in cfg_dict.keys():
        cfg = cfg_dict[cfg_name]
        vertices, elements, elementlabels = cfg["mesh"]
        kernel = cfg["kernel"]
        load = cfg["load"]
        conf = cfg["conf"]

        mesh, A = stiffnessMatrix_fromArray(elements, elementlabels, vertices, kernel, conf)
        slbl = mesh["solutionLabels"]

        A_O = A[slbl > 0][:, slbl > 0]
        A_D = A[slbl > 0][:, slbl < 0]

        if conf["ansatz"] == "CG":
            dirichlet_Vdx = mesh["vertexLabels"] < 0
        if conf["ansatz"] == "DG":
            dirichlet_Vdx = (mesh["elements"][mesh["elementLabels"] < 0]).ravel()
        g = np.apply_along_axis(load["dirichlet"], 1, vertices[dirichlet_Vdx]).ravel()

        f = loadVector(mesh, load, conf)
        #f[slbl == 2] = 0.0
        f_tilde = f[slbl > 0] - A_D.dot(g)

        u = np.zeros(mesh["K"])
        u[slbl > 0] = spsolve(A_O, f_tilde)
        u[slbl < 0] = g

        mesh, u, f = trim_artificial_vertex(mesh, u, f)

        saveplot("plots/"+cfg_name+".pdf", mesh, u, f)
