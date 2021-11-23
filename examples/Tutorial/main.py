import numpy as np
from scipy.sparse.linalg import spsolve
from nlfem import loadVector, stiffnessMatrix_fromArray
from mesh import trim_artificial_vertex
from plot import saveplot
from conf import cfg_list

if __name__ == "__main__":
    for cfg in cfg_list:
        vertices, elements, elementlabels = cfg["mesh"]
        kernel = cfg["kernel"]
        load = cfg["load"]
        conf = cfg["conf"]

        mesh, A = stiffnessMatrix_fromArray(elements, elementlabels, vertices, kernel, conf)
        ndlb = mesh["nodeLabels"]

        A_O = A[ndlb > 0][:, ndlb > 0]
        A_D = A[ndlb > 0][:, ndlb < 0]

        g = np.apply_along_axis(load["dirichlet"], 1, vertices[mesh["vertexLabels"] < 0]).ravel()

        f = loadVector(mesh, load, conf)
        #f[ndlb == 2] = 0.0
        f_tilde = f[ndlb > 0] - A_D.dot(g)

        u = np.zeros(mesh["nV"]*mesh["outdim"])
        u[ndlb > 0] = spsolve(A_O, f_tilde)
        u[ndlb < 0] = g

        mesh, u, f = trim_artificial_vertex(mesh, u, f)

        saveplot("plots/"+cfg["name"]+".pdf", mesh, u, f)
