from time import time
import os
import subprocess
import nlfem
import helpers
import numpy as np
from mesh import RegMesh2D
from scipy.sparse.linalg import cg
from matplotlib.backends.backend_pdf import PdfPages

def runTest(conf, kernel, load, layerDepth, pp = None):
    err_ = None
    data = {"h": [], "L2 Error": [], "Rates": [], "Assembly Time": [], "nV_Omega": []}
    u_exact = load["solution"]

    # Delta is assumed to be of the form deltaK/10 in the mesh, so we obtain deltaK by
    deltaK = int(np.round(kernel["horizon"] * 10))
    if not deltaK:
        raise ValueError("Delta has to be of the form delta = deltaK/10. for deltaK in N.")
    n_start = 10 + 2*deltaK
    N = [n_start * 2 ** l for l in list(range(layerDepth))]
    N_fine = N[-1]*4

    for n in N:
        mesh = RegMesh2D(kernel["horizon"], n, ufunc=u_exact,
                         ansatz=conf["ansatz"], outdim=kernel["outputdim"])
        print("\n h: ", mesh.h)
        data["h"].append(mesh.h)
        data["nV_Omega"].append(mesh.nV_Omega)

        # Assembly ------------------------------------------------------------------------
        start = time()
        A = nlfem.stiffnessMatrix(mesh.__dict__, kernel, conf)
        f = nlfem.loadVector(mesh.__dict__, load, conf)

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
        if kernel["outputdim"] == 1:
            mesh.plot_ud(pp)
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
        Mu_udiff = nlfem.evaluateMass(mesh, u_diff,
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
    import argparse

    parser = argparse.ArgumentParser(description='Run convergence test for the given configuration file.')
    parser.add_argument('-f', default="testConfFull", type=str, help='Enter here the filename of test configuration python file.')
    parser.add_argument('-s', default=1, type=int, help='Number of steps of the convergence test.')
    args = parser.parse_args()
    testFilename = str(args.f)
    layerDepth = int(args.s)

    print("\n### TESTING "+testFilename+"\n")
    if testFilename[-3:] != ".py":
        testFilename += ".py"
    os.system("cp " + testFilename + " testConfiguration.py")
    from testConfiguration import CONFIGURATIONS, KERNELS, LOADS

    os.makedirs("results", exist_ok=True)
    pp = PdfPages("results/plots.pdf")
    tmpstmp = helpers.timestamp()
    fileHandle = open("results/rates" + tmpstmp + ".md", "w+")

    for k, kernel in enumerate(KERNELS):
        load = LOADS[k]
        fileHandle.write("# Kernel: " + kernel["function"] + "\n")
        for conf in CONFIGURATIONS:
            data = runTest(conf, kernel, load, layerDepth, pp)
            helpers.append_output(data, conf, kernel, load, fileHandle=fileHandle)
    fileHandle.close()
    pp.close()
    subprocess.run(f"pandoc results/rates{tmpstmp}.md -o results/rates{tmpstmp}.pdf", shell=True)
    print("\n* Diff of results ********************************************************")
    subprocess.run(f"diff results/rates{tmpstmp}.md results/result_{testFilename[:-3]}.md", shell=True)
    print("**************************************************************************")