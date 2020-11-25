from mesh import RegMesh2D
import helpers
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import nlfem as assemble
from time import time
from scipy.sparse.linalg import cg


def main():
    import conf
    err_ = None
    pp = PdfPages(conf.fnames["triPlot.pdf"])
    for n in conf.N:
        mesh=RegMesh2D(conf.delta, n, ufunc=conf.u_exact, ansatz=conf.ansatz)
        print("\n h: ", mesh.h)
        conf.data["h"].append(mesh.h)
        conf.data["nV_Omega"].append(mesh.nV_Omega)
        mesh.save("data")
        conf.save("data")

        tg = assemble.tensorgauss(4)
        # Assembly ------------------------------------------------------------------------
        start = time()
        A, f = assemble.assemble(mesh, conf.py_Px, conf.py_Py, conf.dx, conf.dy, conf.delta,
                                 model_kernel=conf.model_kernel,
                                 model_f=conf.model_f,
                                 integration_method=conf.integration_method,
                                 is_PlacePointOnCap=conf.is_PlacePointOnCap,
                                 compute="systemforcing",
                                 tensorGaussDegree=conf.tensorGaussDegree)

        conf.data["Assembly Time"].append(time() - start)

        A_O = A[:, :mesh.K_Omega]
        A_I = A[:, mesh.K_Omega:]

        if conf.ansatz == "CG":
            g = np.apply_along_axis(conf.u_exact, 1, mesh.vertices[mesh.nV_Omega:])
        else:
            g = np.zeros(((mesh.K - mesh.K_Omega) // mesh.outdim, mesh.outdim))
            for i, E in enumerate(mesh.elements[mesh.nE_Omega:]):
                for ii, Vdx in enumerate(E):
                    vert = mesh.vertices[Vdx]
                    g[3*i + ii] = conf.u_exact(vert)
        f -= A_I @ g.ravel()

        # Solve ---------------------------------------------------------------------------
        print("Solve...")
        #mesh.write_ud(np.linalg.solve(A_O, f), conf.u_exact)
        x = cg(A_O, f, f)[0].reshape((-1, mesh.outdim))
        #print("CG Solve:\nIterations: ", solution["its"], "\tError: ", solution["res"])
        mesh.write_ud(x, conf.u_exact)

        # Some random quick Check....
        #filter = np.array(assemble.read_arma_mat("data/result.fd").flatten(), dtype=bool)
        #plt.scatter(mesh.vertices[filter][:,0], mesh.vertices[filter][:,1])
        #plt.scatter(mesh.vertices[np.invert(filter)][:,0], mesh.vertices[np.invert(filter)][:,1])
        #plt.show()

        # Refine to N_fine ----------------------------------------------------------------
        mesh.plot_ud(pp)
        mesh = RegMesh2D(conf.delta, conf.N_fine, ufunc=conf.u_exact, coarseMesh=mesh,
                         is_constructAdjaciencyGraph=False, ansatz=conf.ansatz)

        # Evaluate L2 Error ---------------------------------------------------------------
        u_diff = (mesh.u_exact - mesh.ud)[:mesh.K_Omega]
        Mu_udiff = assemble.evaluateMass(mesh, u_diff, conf.py_Px, conf.dx)
        err = np.sqrt(u_diff.ravel() @ Mu_udiff)

        # Print Rates ---------------------------------------------------------------------
        print("L2 Error: ", err)
        conf.data["L2 Error"].append(err)
        if err_ is not None:
            rate = np.log2(err_/err)
            print("Rate: \t",  rate)
            conf.data["Rates"].append(rate)
        else:
            conf.data["Rates"].append(0)
        err_ = err
    pp.close()

    return conf.data

if __name__ == "__main__":
    data = main()
    helpers.write_output(data)