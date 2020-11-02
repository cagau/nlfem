from time import time

import os
import helpers
import nlfem as assemble
import numpy as np
from mesh import RegMesh2D
from scipy.sparse.linalg import cg
from Configuration import u_exact, CONFIGURATIONS, KERNELS, load, tensorGaussDegree


def main(conf, kernel):
    err_ = None
    # pp = PdfPages(conf.fnames["triPlot.pdf"])
    data = {"h": [], "L2 Error": [], "Rates": [], "Assembly Time": [], "nV_Omega": []}

    n_start = 12
    n_layers = 4
    N = [n_start * 2 ** l for l in list(range(n_layers))]
    N_fine = N[-1]*4

    for n in N:
        mesh = RegMesh2D(kernel["horizon"], n, ufunc=u_exact)
        print("\n h: ", mesh.h)
        data["h"].append(mesh.h)
        data["nV_Omega"].append(mesh.nV_Omega)

        # Assembly ------------------------------------------------------------------------
        start = time()
        A, f = assemble.assemble(mesh,
                                 conf["quadrature"]["outer"]["points"],
                                 conf["quadrature"]["inner"]["points"],
                                 conf["quadrature"]["outer"]["weights"],
                                 conf["quadrature"]["inner"]["weights"],
                                 kernel["horizon"],
                                 model_kernel=kernel["function"],
                                 model_f=load["function"],
                                 integration_method=conf["approxBalls"]["method"],
                                 is_PlacePointOnCap=conf["approxBalls"]["isPlacePointOnCap"],
                                 compute="systemforcing",
                                 tensorGaussDegree=tensorGaussDegree)

        data["Assembly Time"].append(time() - start)

        A_O = A[:, :mesh.K_Omega]
        A_I = A[:, mesh.K_Omega:]

        g = np.apply_along_axis(u_exact, 1, mesh.vertices[mesh.K_Omega:])
        f -= A_I @ g

        # Solve ---------------------------------------------------------------------------
        print("Solve...")
        # mesh.write_ud(np.linalg.solve(A_O, f), conf.u_exact)
        x = cg(A_O, f, f)[0]
        # print("CG Solve:\nIterations: ", solution["its"], "\tError: ", solution["res"])
        mesh.write_ud(x, u_exact)

        # Some random quick Check....
        # filter = np.array(assemble.read_arma_mat("data/result.fd").flatten(), dtype=bool)
        # plt.scatter(mesh.vertices[filter][:,0], mesh.vertices[filter][:,1])
        # plt.scatter(mesh.vertices[np.invert(filter)][:,0], mesh.vertices[np.invert(filter)][:,1])
        # plt.show()

        # Refine to N_fine ----------------------------------------------------------------

        mesh = RegMesh2D(kernel["horizon"], N_fine, ufunc=u_exact, coarseMesh=mesh,
                         is_constructAdjaciencyGraph=False)

        # Evaluate L2 Error ---------------------------------------------------------------
        u_diff = (mesh.u_exact - mesh.ud)[:mesh.K_Omega]
        Mu_udiff = assemble.evaluateMass(mesh, u_diff,
                                         conf["quadrature"]["outer"]["points"],
                                         conf["quadrature"]["outer"]["weights"])
        err = np.sqrt(u_diff @ Mu_udiff)

        # Print Rates ---------------------------------------------------------------------
        print("L2 Error: ", err)
        data["L2 Error"].append(err)
        if err_ is not None:
            rate = np.log2(err_ / err)
            print("Rate: \t", rate)
            data["Rates"].append(rate)
        else:
            data["Rates"].append(0)
        err_ = err
    return data


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    fileHandle = open("results/rates.md", "w+")
    for kernel in KERNELS:
        fileHandle.write("# Kernel: " + kernel["function"] + "\n")
        for conf in CONFIGURATIONS:
            data = main(conf, kernel)
            helpers.append_output(data, conf, kernel, load, fileHandle=fileHandle)
    fileHandle.close()
    os.system("pandoc results/rates.md -o results/rates.pdf")