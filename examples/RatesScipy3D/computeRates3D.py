from examples.RatesScipy3D.mesh import RegMesh2D as RegMesh
import examples.RatesScipy3D.helpers as helpers
import numpy as np
import assemble
from time import time
import matplotlib.pyplot as plt
def compareRHS():
    import examples.RatesScipy3D.conf3D as conf
    dim = 3
    n = conf.n_start
    def source(x):
        return 1.#2.*(x[1] + 2.)

    mesh = RegMesh(conf.delta, n, dim=3, boundaryConditionType="Neumann")
    # Assembly ------------------------------------------------------------------------
    A, ffromAssemble = assemble.assemble(mesh, conf.py_Px, conf.py_Py, conf.dx, conf.dy, conf.delta,
                             model_kernel=conf.model_kernel,
                             model_f=conf.model_f,
                             integration_method=conf.integration_method,
                             is_PlacePointOnCap=conf.is_PlacePointOnCap)

    ffromMass_K  = np.zeros(mesh.K)
    sourceData = np.array([source(v) for v in mesh.vertices])


    if mesh.is_NeumannBoundary:
        ffromAssemble_K = ffromAssemble

        ffromMass_K = assemble.evaluateMass(mesh, sourceData, conf.py_Px, conf.dx)
    else:
        ffromAssemble_K = np.zeros(mesh.K)
        ffromAssemble_K[:mesh.K_Omega] = ffromAssemble

        ffromMass_K[:mesh.K_Omega] = assemble.evaluateMass(mesh, sourceData, conf.py_Px, conf.dx)


    mesh.plot3D(conf.fnames["tetPlot.vtk"], dataDict={"ffromAssemble": ffromAssemble_K, "ffromMass": ffromMass_K})

def rates():
    import examples.RatesScipy3D.conf3D as conf
    err_ = None
    dim = 3
    for n in conf.N:
        print()
        mesh = RegMesh(conf.delta, n, dim=3)
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
        #plt.imsave("results/A.pdf", A )
        #raise KeyboardInterrupt
        A_O = A[:,:mesh.K_Omega]
        A_I = A[:,mesh.K_Omega:]
        g = np.apply_along_axis(conf.u_exact, 1, mesh.vertices[mesh.K_Omega:])
        f -= A_I@g

        # Solve ---------------------------------------------------------------------------
        print("Solve...")
        #mesh.write_ud(np.linalg.solve(A_O, f), conf.u_exact)
        solution = assemble.solve_cg(A_O, f, f)
        print("CG Solve:\nIterations: ", solution["its"], "\tError: ", solution["res"])
        mesh.write_ud(solution["x"], conf.u_exact)

        # Refine to N_fine ----------------------------------------------------------------
        mesh = RegMesh(conf.delta, conf.N_fine, coarseMesh=mesh, ufunc=conf.u_exact, dim=3, is_constructAdjaciencyGraph=False)

        # Evaluate L2 Error ---------------------------------------------------------------

        u_diff = (mesh.u_exact - mesh.ud)[:mesh.K_Omega]
        Mu_udiff = assemble.evaluateMass(mesh, u_diff, conf.py_Px, conf.dx)
        err = np.sqrt(u_diff @ Mu_udiff)
        mesh.plot3D(conf.fnames["tetPlot.vtk"])
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
    return conf.data



def main():
    import examples.RatesScipy3D.conf3D as conf
    u_exact = lambda x: (x[0]-0.5)**2 + (x[1]-0.5)**2 + (x[2]-0.5)**2 - 0.75
    err_ = None
    dim = 3
    conf.delta = 0.5
    conf.n_start = 4
    mesh = RegMesh(conf.delta,conf.n_start, dim=3, ufunc=u_exact)
    conf.data["h"].append(mesh.h)
    conf.data["nV_Omega"].append(mesh.nV_Omega)
    conf.save("data")
    mesh.save("data")
    mesh.plot3D("results/plot3D.vtk")



    def det(Vdx):
        T = mesh.vertices[Vdx]
        a,b,c,d = T
        M = np.array([b-a, c-a, d-a])
        Tdet = np.linalg.det(M)
        return Tdet
    TDets  = np.array([det(T) for T in mesh.elements])
    print(TDets)
    print("N Zeros: ", np.sum(TDets == 0.0))
    # Assembly ------------------------------------------------------------------------
    start = time()
    A, f = assemble.assemble(mesh, conf.py_Px, conf.py_Py, conf.dx, conf.dy, conf.delta,
                             model_kernel=conf.model_kernel,
                             model_f=conf.model_f,
                             integration_method=conf.integration_method,
                             is_PlacePointOnCap=conf.is_PlacePointOnCap)
    conf.data["Assembly Time"].append(time() - start)
    #mesh.plot3D(conf.fnames["tetPlot.vtk"])

if __name__ == "__main__":
    #main()
    data = rates()
    #helpers.write_output(data)
    #compareRHS()