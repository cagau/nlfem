from time import time

import os
import helpers
import nlfem as assemble
import numpy as np
from mesh import RegMesh2D
from scipy.sparse.linalg import cg
from matplotlib.backends.backend_pdf import PdfPages

def main(conf, kernel, load, pp = None):
    err_ = None
    data = {"h": [], "L2 Error": [], "Rates": [], "Assembly Time": [], "nV_Omega": []}
    u_exact = load["solution"]

    n_start = 12
    n_layers = 3
    N = [n_start * 2 ** l for l in list(range(n_layers))]
    N_fine = N[-1]*4

    for n in N:
        mesh = RegMesh2D(kernel["horizon"], n, ufunc=u_exact,
                         ansatz = conf["ansatz"], outdim=kernel["outputdim"])
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
                                 tensorGaussDegree=conf["quadrature"]["tensorGaussDegree"])

        data["Assembly Time"].append(time() - start)

        A_O = A[:, :mesh.K_Omega]
        A_I = A[:, mesh.K_Omega:]

        if conf["ansatz"] == "CG":
            g = np.apply_along_axis(u_exact, 1, mesh.vertices[mesh.nV_Omega:])
        else:
            g = np.zeros(((mesh.K - mesh.K_Omega) // mesh.outdim, mesh.outdim))
            for i, E in enumerate(mesh.elements[mesh.nE_Omega:]):
                for ii, Vdx in enumerate(E):
                    vert = mesh.vertices[Vdx]
                    g[3*i + ii] = u_exact(vert)
        f -= A_I @ g.ravel()

        # Solve ---------------------------------------------------------------------------
        print("Solve...")
        # mesh.write_ud(np.linalg.solve(A_O, f), conf.u_exact)
        x = cg(A_O, f, f)[0].reshape((-1, mesh.outdim))
        # print("CG Solve:\nIterations: ", solution["its"], "\tError: ", solution["res"])
        mesh.write_ud(x, u_exact)
        #mesh.plot_ud(pp)
        # Some random quick Check....
        # filter = np.array(assemble.read_arma_mat("data/result.fd").flatten(), dtype=bool)
        # plt.scatter(mesh.vertices[filter][:,0], mesh.vertices[filter][:,1])
        # plt.scatter(mesh.vertices[np.invert(filter)][:,0], mesh.vertices[np.invert(filter)][:,1])
        # plt.show()

        # Refine to N_fine ----------------------------------------------------------------
        mesh = RegMesh2D(kernel["horizon"], N_fine,
                         ufunc=u_exact, coarseMesh=mesh,
                         is_constructAdjaciencyGraph=False,
                         ansatz=conf["ansatz"], outdim=kernel["outputdim"])
        #mesh.plot_ud(pp)
        #mesh.plot_u_exact(pp)
        # Evaluate L2 Error ---------------------------------------------------------------
        u_diff = (mesh.u_exact - mesh.ud)[:(mesh.K_Omega // mesh.outdim)]
        Mu_udiff = assemble.evaluateMass(mesh, u_diff,
                                         conf["quadrature"]["outer"]["points"],
                                         conf["quadrature"]["outer"]["weights"])
        err = np.sqrt(u_diff.ravel() @ Mu_udiff)

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
    #pp.close()
    return data

if __name__ == "__main__":
    from testConfFull import CONFIGURATIONS, KERNELS, LOADS
    #from testConfPeridyn import CONFIGURATIONS, KERNELS, LOADS
    #from testConfConstant import CONFIGURATIONS, KERNELS, LOADS
    #from testConfWCCM1 import CONFIGURATIONS, KERNELS, LOADS
    #from testConfWCCM2 import CONFIGURATIONS, KERNELS, LOADS

    pp = PdfPages("results/plots.pdf")
    os.makedirs("results", exist_ok=True)
    tmpstmp = helpers.timestamp()
    fileHandle = open("results/rates" + tmpstmp + ".md", "w+")
    for k, kernel in enumerate(KERNELS):
        load = LOADS[k]
        fileHandle.write("# Kernel: " + kernel["function"] + "\n")
        for conf in CONFIGURATIONS:
            data = main(conf, kernel, load, pp)
            helpers.append_output(data, conf, kernel, load, fileHandle=fileHandle)
    fileHandle.close()
    pp.close()
    os.system("pandoc results/rates" + tmpstmp + ".md -o results/rates" + tmpstmp + ".pdf")