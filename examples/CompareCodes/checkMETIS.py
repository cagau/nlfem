import matplotlib.pyplot as plt
import nlfem
import nlfem as assemble
import numpy as np
import os


def bC(mesh):
    baryCenter = np.zeros((mesh.nE, 2))
    for jj, T in enumerate(mesh.elements):
        baryCenter[jj] = np.sum(mesh.vertices[T], axis=0) / 3.
    return baryCenter

if __name__ == "__main__":
    from mesh import RegMesh2D
    import conf
    mesh = RegMesh2D(conf.delta, 48)

    # Write
    print("\n h: ", mesh.h)
    conf.data["h"].append(mesh.h)
    conf.data["nV_Omega"].append(mesh.nV_Omega)
    mesh.save("data")
    conf.save("data")

    # Run METIS
    # (Führt C++ Code aus, nämlich die main() Funktion in comparecodes.cpp.
    # Da ist eigentlich nur die Funktion readConfiguration() interessant dort wird am Ende METIS aufgerufen.
    # Das Ergebnis wird in eine File gespeichert: data/result.partition)
    #                         v Anzahl der Subdomains
    os.system("./CompareCodes 6")

    # Read
    filter = np.array(assemble.read_arma_mat("data/result.partition").flatten(), dtype=np.int)
    adjacency = np.array(assemble.read_arma_mat("data/result.dual").flatten(), dtype=np.int).reshape((50, mesh.nE)).T
    interaction = assemble.read_arma_spMat("data/result.interaction")
    baryCenters = bC(mesh)
    adjacency = np.sort(adjacency, axis=1)
    adjacency_test = nlfem.constructAdjaciencyGraph(mesh.elements, ncommon=1)
    for k in range(mesh.nE):
        l = np.where(adjacency_test[k] == k)
        adjacency_test[k, l] = mesh.nE
    adjacency_test = np.sort(adjacency_test, axis=1)
    diff = adjacency[:, :14] - adjacency_test[:, :14]
    domains = np.zeros((filter.shape[0], np.max(filter)+1), dtype=np.int)
    for k, f in enumerate(filter):
        domains[k, f] = 1

    # Plot
    labels = np.unique(filter)
    for l in labels:
        filter_l = filter == l
        plt.scatter(mesh.vertices[filter_l][:,0], mesh.vertices[filter_l][:,1])
    plt.show()


