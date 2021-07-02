from time import time
import os
import subprocess
import nlfem
import helpers
import numpy as np
from mesh import RegMesh2D
from scipy.sparse.linalg import cg
from matplotlib.backends.backend_pdf import PdfPages

class Callback:
    def __init__(self):
        self.n = 0
    def __call__(self, *args, **kwargs):
        self.n += 1

def runTest(conf, kernel, load, layerDepth, pp = None):
    err_ = None
    data = {"$h$": [], "$K_\Omega$": [], "$\delta$": [], "L2 Error": [], "Rates": [], "Time [s]": []}
    u_exact = load["solution"]

    # Delta is assumed to be of the form deltaK/10 in the mesh, so we obtain deltaK by
    # This is a restriction due to the simplified mesh-generation and independent of the
    # assembly routine!
    n_start = 5 + 4
    N_fine = 2**(layerDepth+2)
    #N = [n_start * 2 ** (layerDepth-1)]*layerDepth
    N = [n_start * 2 ** n for n in range(0, layerDepth)]
    Delta = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625] #[0.1, 0.05, 0.025, 0.0125, 0.0125]


    for k, delta in enumerate(Delta[:layerDepth]):
        kernel["horizon"] = delta
        data["$\delta$"].append(delta)
        mesh = RegMesh2D(0.2, N[k], ufunc=u_exact,
                         ansatz=conf["ansatz"], outdim=kernel["outputdim"], variant="down")
        mesh.delta = delta
        print("\n h: ", mesh.h)
        data["$h$"].append(mesh.h)
        data["$K_\Omega$"].append(mesh.K_Omega)

        # Assembly ------------------------------------------------------------------------
        start = time()
        A = nlfem.stiffnessMatrix(mesh.__dict__, kernel, conf)
        nnzRows = np.max(np.sum(A != 0, axis=0))
        print("nnzRows are ", nnzRows)
        f_OI = nlfem.loadVector(mesh.__dict__, load, conf)
        data["Time [s]"].append(time() - start)

        A_O = A[mesh.nodeLabels > 0][:, mesh.nodeLabels > 0]
        #A_Odense = np.array(A_O.todense())
        #test =  np.linalg.norm(A_Odense - A_Odense.T)
        #print("########### Sym Check: ", test)

        A_I = A[mesh.nodeLabels > 0][:, mesh.nodeLabels < 0]
        #f_OI[mesh.nodeLabels > 1] *= -10
        f = f_OI[mesh.nodeLabels > 0]

        if conf["ansatz"] == "CG":
            g = np.apply_along_axis(u_exact, 1, mesh.vertices[mesh.vertexLabels < 0])
        else:
            g = np.zeros(((mesh.K - mesh.K_Omega) // mesh.outdim, mesh.outdim))
            for i, E in enumerate(mesh.elements[mesh.elementLabels < 0]):
                for ii, Vdx in enumerate(E):
                    vert = mesh.vertices[Vdx]
                    g[(mesh.dim+1)*i + ii] = u_exact(vert)
        f -= A_I @ g.ravel()

        # Solve ---------------------------------------------------------------------------
        print("Solve...")
        # mesh.write_ud(np.linalg.solve(A_O, f), conf.u_exact)
        cb = Callback()
        x = cg(A_O, f, f, tol=1e-9, callback=cb)[0].reshape((-1, mesh.outdim))
        print("Number of cg iterations: ", cb.n)
        A = None
        A_O = None
        A_I = None
        # print("CG Solve:\nIterations: ", solution["its"], "\tError: ", solution["res"])
        mesh.write_ud(x, u_exact)
        if kernel["outputdim"] == 1 and mesh.dim == 2:
            mesh.plot_ud(pp, is_quickDG=True)
            #mesh.plot_vertexLabels(pp)
        # Some random quick Check....
        # filter = np.array(assemble.read_arma_mat("data/result.fd").flatten(), dtype=bool)
        # plt.scatter(mesh.vertices[filter][:,0], mesh.vertices[filter][:,1])
        # plt.scatter(mesh.vertices[np.invert(filter)][:,0], mesh.vertices[np.invert(filter)][:,1])
        # plt.show()

        # Refine to N_fine ----------------------------------------------------------------
        mesh = RegMesh2D(0.2, N_fine,
                         ufunc=u_exact, coarseMesh=mesh,
                         ansatz=conf["ansatz"], outdim=kernel["outputdim"], variant="down")
        #mesh.plot_u_exact(pp)
        # Evaluate L2 Error ---------------------------------------------------------------
        u_diff = (mesh.u_exact - mesh.ud)
        Mu_udiff = nlfem.evaluateMass(mesh, u_diff,
                                         conf["quadrature"]["outer"]["points"],
                                         conf["quadrature"]["outer"]["weights"])

        err = np.sqrt(u_diff.ravel() @ Mu_udiff)

        # Print Rates ---------------------------------------------------------------------
        print("L2 Error: ", err)
        data["L2 Error"].append(err)
        if err_ is not None and err_ > 1e-16:
            rate = np.log(err_ / err)/np.log(2)
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
    os.system("cp conf/" + testFilename + " conf/testConfiguration.py")
    from conf.testConfiguration import CONFIGURATIONS, KERNELS, LOADS

    os.makedirs("results", exist_ok=True)
    host = os.uname()[1]
    pp = PdfPages("results/plots.pdf")
    tmpstmp = helpers.timestamp()
    suffix = "_" + host + "_"  + tmpstmp
    fileHandle = open("results/rates" + suffix + ".md", "w+")

    for k, kernel in enumerate(KERNELS):
        load = LOADS[k]
        #fileHandle.write("# Kernel: " + kernel["function"] + "\n")
        for conf in CONFIGURATIONS:
            data = runTest(conf, kernel, load, layerDepth, pp)
            helpers.append_output(data, conf, kernel, load, fileHandle=fileHandle, datacolumns=data)
    fileHandle.close()
    pp.close()
    subprocess.run(f"pandoc results/rates{suffix}.md -o results/rates{suffix}.tex", shell=True)
    subprocess.run(f"pandoc results/rates{suffix}.md -o results/rates{suffix}.pdf", shell=True)
    print("\n* Diff of results ********************************************************")
    subprocess.run(f"diff results/rates{suffix}.md results/result_{testFilename[:-3]}.md", shell=True)
    print("**************************************************************************")