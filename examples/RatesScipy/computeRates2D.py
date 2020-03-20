from examples.RatesScipy.mesh import RegMesh2D
from scipy.spatial.distance import  euclidean as l2dist
import examples.RatesScipy.helpers as helpers
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import assemble
from time import time

def main():
    import examples.RatesScipy.conf as conf
    err_ = None

    mesh_exact = RegMesh2D(conf.delta, conf.N_fine, conf.u_exact)

    pp = PdfPages(conf.fnames["triPlot.pdf"])
    for n in conf.N:
        print()
        mesh=RegMesh2D(conf.delta, n)
        conf.data["h"].append(mesh.h)
        conf.data["nV_Omega"].append(mesh.nV_Omega)

        # Assembly ------------------------------------------------------------------------
        start = time()
        A, f = assemble.assemble(mesh, conf.py_Px, conf.py_Py, conf.dx, conf.dy, conf.delta,
                                 model_kernel=conf.model_kernel,
                                 model_f=conf.model_f,
                                 integration_method=conf.integration_method,
                                 is_PlacePointOnCap=conf.is_PlacePointOnCap)
        conf.data["Assembly Time"].append(time() - start)

        A_O = A[:,:mesh.K_Omega]
        A_I = A[:,mesh.K_Omega:]

        g = np.apply_along_axis(conf.u_exact, 1, mesh.vertices[mesh.K_Omega:])
        f -= A_I@g

        # Solve ---------------------------------------------------------------------------
        print("Solve...")
        mesh.write_u(np.linalg.solve(A_O,f), conf.u_exact)
        mesh.plot(pp)

        # Refine to N_fine ----------------------------------------------------------------
        mesh = RegMesh2D(conf.delta, conf.N_fine, coarseMesh=mesh)

        # Evaluate L2 Error ---------------------------------------------------------------
        u_diff = (mesh_exact.u - mesh.u)[:mesh_exact.K_Omega]
        Mu_udiff = assemble.evaluateMass(mesh_exact, u_diff, conf.py_Px, conf.dx)
        err = np.sqrt(u_diff @ Mu_udiff)

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